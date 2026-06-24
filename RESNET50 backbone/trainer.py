import torch
import math
from mdep_agents import MDEPLinear, MDEPConv2d, update_scores_agents
from edl_core import compute_uncertainties

class MDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        if warmup_epochs is None:
            self.warmup_epochs = max(1, int(0.20 * total_epochs))
        else:
            self.warmup_epochs = warmup_epochs
        
        # User requested parameters for gamma
        self.gamma_initial = 5.0
        self.gamma_final = 0.15
        
        # AMP Scaler for Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()
        
    def step_gamma(self, epoch):
        if epoch < self.warmup_epochs:
            return self.gamma_initial
        # Cosine annealing for Smoothed STE temperature
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        gamma = self.gamma_final + 0.5 * (self.gamma_initial - self.gamma_final) * (1 + math.cos(math.pi * progress))
        return gamma

    def set_warmup_state(self, is_warmup, gamma):
        for module in self.model.modules():
            if isinstance(module, (MDEPLinear, MDEPConv2d)):
                module.warmup = is_warmup
                module.gamma = gamma

    def reset_effective_weight_grads(self):
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if hasattr(m, 'effective_weight') and m.effective_weight is not None:
                    m.effective_weight.grad = None

    def compute_amortized_gradients(self, inputs):
        """
        Amortized backward passes that compute:
          • ∂u_a / ∂w      → signal for the Microglia agent
          • ∂u_e / ∂a^(l)  → signal for the Astrocyte agent (per-neuron)
        Called only once per epoch to keep FLOPs low.
        """
        self.model.train()

        # Register forward hooks to capture layer activations for Astrocyte
        activations = {}
        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                def _hook(module, inp, out, n=name):
                    activations[n] = out
                hooks.append(m.register_forward_hook(_hook))

        outputs = self.model(inputs)
        uncertainties = compute_uncertainties(outputs)

        u_a = torch.mean(uncertainties['aleatoric'])
        # Class-selective epistemic target to resolve gradient blindness
        u_e_target = torch.mean(torch.sum(1.0 / uncertainties['alpha'], dim=-1))

        # 1. ∂u_a/∂w → Microglia agent (per-weight signal)
        self.model.zero_grad()
        self.reset_effective_weight_grads()
        u_a.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if hasattr(m, 'effective_weight') and m.effective_weight.grad is not None:
                    m.grad_ua_w = m.effective_weight.grad.clone().detach()
                else:
                    m.grad_ua_w = torch.zeros_like(m.weight)

        # Clear grads of all parameters and intermediate weight tensors to isolate the u_e graph
        self.model.zero_grad()
        self.reset_effective_weight_grads()

        # 2. ∂u_e/∂a^(l) → Astrocyte agent (per-neuron signal)
        act_tensors = []
        act_modules = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)) and name in activations:
                act_tensors.append(activations[name])
                act_modules.append(m)

        if act_tensors:
            grads = torch.autograd.grad(u_e_target, act_tensors, allow_unused=True)
            for m, grad in zip(act_modules, grads):
                if grad is not None:
                    if isinstance(m, MDEPLinear):
                        m.u_e_node = torch.abs(grad).mean(dim=0).detach()
                    elif isinstance(m, MDEPConv2d):
                        m.u_e_node = torch.abs(grad).mean(dim=(0, 2, 3)).detach()
                else:
                    m.u_e_node = None

        for h in hooks:
            h.remove()
        self.model.zero_grad()
        self.reset_effective_weight_grads()

    def train_epoch(self, epoch, dataloader, device, print_interval=200):
        self.model.train()
        
        is_warmup = epoch < self.warmup_epochs
        gamma = self.step_gamma(epoch)
        self.set_warmup_state(is_warmup, gamma)
        
        # Manual LR Warmup parameters
        warmup_period = 1
        base_lr = 4.0e-05
        num_batches = len(dataloader)
                
        total_loss = 0
        total_grad_norm = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Smooth per-batch LR Warmup
            if epoch < warmup_period:
                current_step = epoch * num_batches + batch_idx
                total_warmup_steps = warmup_period * num_batches
                current_lr = 1e-6 + (base_lr - 1e-6) * (current_step / total_warmup_steps)
                
                # Linear decay for Loss Scaling from 4.0 to 1.0 to prevent overshooting
                current_loss_scale = 4.0 - 3.0 * (current_step / total_warmup_steps)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = base_lr
                current_loss_scale = 1.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            inputs, targets = inputs.to(device), targets.to(device)
            
            # Amortized evaluation on first batch (also on warm-up epoch 0 and 1 for diagnostics)
            if (not is_warmup or epoch < 2) and batch_idx == 0:
                self.compute_amortized_gradients(inputs)
                
            self.model.zero_grad()
            self.reset_effective_weight_grads()
            
            # Use Automatic Mixed Precision for Forward Pass
            with torch.cuda.amp.autocast():
                evidence = self.model(inputs)
            
            # Ensure Evidential Loss runs strictly in FP32 to avoid digamma/log underflow
            with torch.cuda.amp.autocast(enabled=False):
                loss = self.criterion(evidence.float(), targets, epoch)
            
            # Loss scaling to counteract Focal Loss shrinkage (decayed)
            scaled_loss = loss * current_loss_scale
            
            self.scaler.scale(scaled_loss).backward()
            
            # Gradient clipping and norm tracking (only for optimized parameters)
            self.scaler.unscale_(self.optimizer)
            params_to_clip = [p for group in self.optimizer.param_groups for p in group['params']]
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            total_grad_norm += grad_norm.item()
            
            # Cache primary weight gradient for structural updates
            if not is_warmup or epoch < 2:
                inv_scale = 1.0 / (self.scaler.get_scale() + 1e-8)
                for module in self.model.modules():
                    if isinstance(module, (MDEPLinear, MDEPConv2d)):
                        if hasattr(module, 'effective_weight') and module.effective_weight.grad is not None:
                            module.grad_L_w = module.effective_weight.grad.clone().detach() * inv_scale
                        else:
                            module.grad_L_w = torch.zeros_like(module.weight)

            if epoch < 2 and batch_idx == 0:
                self.check_gradient_flow(epoch)
                        
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Perform multi-agent structure optimization
            if not is_warmup and batch_idx == 0:
                update_scores_agents(self.model)
            
            # Reset parameter and intermediate grads to avoid cross-batch contamination
            self.model.zero_grad()
            self.reset_effective_weight_grads()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_grad = total_grad_norm / len(dataloader)
        print(f"    Epoch {epoch} | LR: {current_lr:.2e} | GradNorm: {avg_grad:.4f}")
        return avg_loss

    def check_gradient_flow(self, epoch):
        import os
        import matplotlib.pyplot as plt
        
        # Create artifacts folder if not exists
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        target_layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                target_layers.append((name, m))
                
        if not target_layers:
            print("No MDEP layers found to check gradient flow.")
            return
            
        print(f"\n[INFO] [Gradient Flow Check - Epoch {epoch}]")
        print("-" * 75)
        
        # We will check the last layer
        visualize_layers = [target_layers[-1]]
        
        for name, m in visualize_layers:
            w_val = m.weight.data.cpu()
            grad_ua = getattr(m, 'grad_ua_w', None)
            grad_L = getattr(m, 'grad_L_w', None)
            
            if grad_ua is None or grad_L is None:
                print(f"Layer {name}: grad_ua_w or grad_L_w is None. Cannot perform flow check.")
                continue
                
            grad_ua = grad_ua.cpu()
            grad_L = grad_L.cpu()
            
            # Magnitudes in Taylor weight-space: |w * grad_w|
            mag_ua = torch.abs(w_val * grad_ua)
            mag_L = torch.abs(w_val * grad_L)
            
            # Min-Max Normalization (Strategy 1)
            mag_ua_min = mag_ua.min()
            mag_ua_max = mag_ua.max()
            mag_ua_norm = (mag_ua - mag_ua_min) / (mag_ua_max - mag_ua_min + 1e-8)
            
            mag_L_min = mag_L.min()
            mag_L_max = mag_L.max()
            mag_L_norm = (mag_L - mag_L_min) / (mag_L_max - mag_L_min + 1e-8)
            
            # Print raw statistics
            print(f"Layer: {name}")
            print(f"  |w * du_a/dw| (Raw): mean={mag_ua.mean().item():.2e}, std={mag_ua.std().item():.2e}, max={mag_ua.max().item():.2e}")
            print(f"  |w * dL_EFL/dw| (Raw): mean={mag_L.mean().item():.2e}, std={mag_L.std().item():.2e}, max={mag_L.max().item():.2e}")
            
            # Relative scale check (beta balance)
            ratio = mag_ua.mean() / (mag_L.mean() + 1e-8)
            print(f"  Ratio (Raw) |w * du_a/dw| / |w * dL/dw|: {ratio.item():.4f}")
            
            # Normalized statistics
            ratio_norm = mag_ua_norm.mean() / (mag_L_norm.mean() + 1e-8)
            print(f"  Normalized |w * du_a/dw|: mean={mag_ua_norm.mean().item():.4f}, std={mag_ua_norm.std().item():.4f}")
            print(f"  Normalized |w * dL_EFL/dw|: mean={mag_L_norm.mean().item():.4f}, std={mag_L_norm.std().item():.4f}")
            print(f"  Ratio (Normalized): {ratio_norm.item():.4f}")
            
            # Plot histograms
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Flatten to 1D
            mag_ua_flat = mag_ua.numpy().flatten()
            mag_L_flat = mag_L.numpy().flatten()
            
            ax1.hist(mag_ua_flat, bins=50, color='blue', alpha=0.7)
            ax1.set_title(f'|w * du_a/dw| ({name})')
            ax1.set_xlabel('Magnitude')
            ax1.set_ylabel('Count')
            
            ax2.hist(mag_L_flat, bins=50, color='green', alpha=0.7)
            ax2.set_title(f'|w * dL_EFL/dw| ({name})')
            ax2.set_xlabel('Magnitude')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            plot_path = os.path.join(artifacts_dir, f"grad_ua_flow_epoch_{epoch}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Saved gradient histogram to: {plot_path}")
            print("-" * 75)
