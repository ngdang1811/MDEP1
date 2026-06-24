import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from swin_agents import MDEPLinear, MDEPConv2d, update_scores_agents
from swin_mdep import compute_uncertainties

def dirichlet_kl_divergence(alpha_s, alpha_t):
    """
    Computes the Kullback-Leibler divergence between two Dirichlet distributions
    represented by concentration parameters alpha_s (student) and alpha_t (teacher).
    """
    S_s = torch.sum(alpha_s, dim=-1, keepdim=True)
    S_t = torch.sum(alpha_t, dim=-1, keepdim=True)
    
    ln_s = torch.lgamma(S_s) - torch.sum(torch.lgamma(alpha_s), dim=-1, keepdim=True)
    ln_t = torch.lgamma(S_t) - torch.sum(torch.lgamma(alpha_t), dim=-1, keepdim=True)
    
    dg_s = torch.digamma(alpha_s)
    dg_S = torch.digamma(S_s)
    
    diff = alpha_s - alpha_t
    term = torch.sum(diff * (dg_s - dg_S), dim=-1, keepdim=True)
    
    kl = ln_s - ln_t + term
    return torch.mean(kl)

class SwinMDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs, 
                 teacher_model=None, alpha_d_initial=0.5, alpha_d_final=0.05):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.teacher_model = teacher_model
        
        # Distillation scheduling parameters
        self.alpha_d_initial = alpha_d_initial
        self.alpha_d_final = alpha_d_final
        
        # Smoothed-STE exploration temperature scheduling
        self.gamma_initial = 5.0
        self.gamma_final = 0.15
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.last_flop_rate = 0.0

    def step_alpha_d(self, epoch):
        """
        Cosine-annealed distillation weight alpha_d(t).
        Early stage (during warm-up): high distillation (0.5) to anchor the student topology.
        Later stage (after warm-up): decays to 0.05 to let the student self-optimize on true labels.
        """
        if epoch < self.warmup_epochs:
            return self.alpha_d_initial
        
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        alpha_d = self.alpha_d_final + 0.5 * (self.alpha_d_initial - self.alpha_d_final) * (
            1.0 + math.cos(math.pi * progress)
        )
        return alpha_d

    def step_gamma(self, epoch):
        """Cosine-annealed temperature for the Smoothed STE."""
        if epoch < self.warmup_epochs:
            return self.gamma_initial
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        gamma = self.gamma_final + 0.5 * (self.gamma_initial - self.gamma_final) * (
            1.0 + math.cos(math.pi * progress)
        )
        return gamma

    def set_warmup_state(self, is_warmup, gamma):
        for module in self.model.modules():
            if isinstance(module, (MDEPLinear, MDEPConv2d)):
                if module.warmup and not is_warmup:
                    from swin_agents import generate_2_4_mask
                    raw_mask = generate_2_4_mask(module.scores.data)
                    module.mask.copy_(raw_mask)
                    if module.scores.numel() % 4 == 0:
                        scores_flat = module.scores.data.view(-1, 4)
                        sorted_scores, _ = torch.sort(scores_flat, dim=-1, descending=True)
                        s2 = sorted_scores[:, 1]
                        s3 = sorted_scores[:, 2]
                        tau = ((s2 + s3) / 2.0).unsqueeze(-1).expand_as(scores_flat).reshape(module.scores.shape)
                        module.tau.copy_(tau)
                    else:
                        module.tau.zero_()
                module.warmup = is_warmup
                module.gamma = gamma

    def reset_effective_weight_grads(self):
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if hasattr(m, 'effective_weight') and m.effective_weight is not None:
                    m.effective_weight.grad = None

    def compute_amortized_gradients(self, inputs):
        """
        Amortized backward passes to calculate:
          • ∂u_a / ∂w      → signal for the Microglia agent (per-weight)
          • ∂u_e / ∂a^(l)  → signal for the Astrocyte agent (per-neuron, averaged over spatial dimensions)
        Called once per epoch to minimize computational overhead.
        """
        self.model.train()

        # Register forward hooks to capture layer output activations
        activations = {}
        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                def _hook(module, inp, out, n=name):
                    activations[n] = out
                hooks.append(m.register_forward_hook(_hook))

        # Forward pass on a batch of inputs
        outputs = self.model(inputs)
        uncertainties = compute_uncertainties(outputs)

        u_a = torch.mean(uncertainties['aleatoric'])
        # Class-selective epistemic target to resolve gradient blindness
        u_e_target = torch.mean(torch.sum(1.0 / uncertainties['alpha'], dim=-1))

        # 1. ∂u_a/∂w → Microglia Agent (noise evaluation signal)
        self.model.zero_grad()
        self.reset_effective_weight_grads()
        u_a.backward(retain_graph=True)
        
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if hasattr(m, 'effective_weight') and m.effective_weight.grad is not None:
                    m.grad_ua_w = m.effective_weight.grad.clone().detach()
                else:
                    m.grad_ua_w = torch.zeros_like(m.weight)

        self.model.zero_grad()
        self.reset_effective_weight_grads()

        # 2. ∂u_e/∂a^(l) → Astrocyte Agent (uncertainty propagation signal)
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
                        # General reduction logic: mean over all dimensions except the final channel dimension (D)
                        # This naturally supports 2D (FC), 3D (tokens), and 4D (Swin spatial grids) activations.
                        m.u_e_node = torch.abs(grad).mean(dim=list(range(grad.dim() - 1))).detach()
                    elif isinstance(m, MDEPConv2d):
                        # mean over batch (0) and spatial dimensions (2, 3)
                        m.u_e_node = torch.abs(grad).mean(dim=(0, 2, 3)).detach()
                else:
                    m.u_e_node = None

        # Clean up registered hooks to prevent memory leaks
        for h in hooks:
            h.remove()
            
        self.model.zero_grad()
        self.reset_effective_weight_grads()

    def check_gradient_flow(self, epoch):
        """Diagnostic tool to inspect structural and task gradients flow on the final MDEP layer."""
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        target_layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                target_layers.append((name, m))
                
        if not target_layers:
            return
            
        name, m = target_layers[-1]
        w_val = m.weight.data.cpu()
        grad_ua = getattr(m, 'grad_ua_w', None)
        grad_L = getattr(m, 'grad_L_w', None)
        
        if grad_ua is None or grad_L is None:
            return
            
        grad_ua = grad_ua.cpu()
        grad_L = grad_L.cpu()
        
        mag_ua = torch.abs(w_val * grad_ua)
        mag_L = torch.abs(w_val * grad_L)
        
        print(f"\n[Diagnostics - Gradient Flow check: {name}]")
        print(f"  |w * du_a/dw| (Microglia Noise): mean={mag_ua.mean().item():.2e}, std={mag_ua.std().item():.2e}")
        print(f"  |w * dL_EFL/dw| (Task Classify): mean={mag_L.mean().item():.2e}, std={mag_L.std().item():.2e}")
        
        # Save analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.hist(mag_ua.numpy().flatten(), bins=50, color='royalblue', alpha=0.8)
        ax1.set_title("Microglia Noise Gradient")
        ax2.hist(mag_L.numpy().flatten(), bins=50, color='forestgreen', alpha=0.8)
        ax2.set_title("Task Loss Gradient")
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, f"grad_ua_flow_epoch_{epoch}.png"))
        plt.close()

    def train_epoch(self, epoch, dataloader, device, print_interval=100, scheduler=None):
        self.model.train()
        
        is_warmup = epoch < self.warmup_epochs
        gamma = self.step_gamma(epoch)
        alpha_d = self.step_alpha_d(epoch)
        self.set_warmup_state(is_warmup, gamma)
        
        base_lr = 2.0e-5
        warmup_period = 1
        num_batches = len(dataloader)
        epoch_start = time.time()
        
        ema_loss = None
        ema_grad = None
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Batch learning rate warmup during first epoch to avoid sudden gradient explosion
            if epoch < warmup_period:
                current_step = epoch * num_batches + batch_idx
                total_warmup_steps = warmup_period * num_batches
                current_lr = 1e-6 + (base_lr - 1e-6) * (current_step / total_warmup_steps)
                current_loss_scale = 4.0 - 3.0 * (current_step / total_warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_loss_scale = 1.0
                if scheduler is None:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = base_lr
                current_lr = self.optimizer.param_groups[0]['lr']

            inputs, targets = inputs.to(device), targets.to(device)

            update_interval = 10
            is_update_step = (not is_warmup or epoch < 2) and ((batch_idx % update_interval == 0) or (batch_idx == num_batches - 1))

            # Compute amortized structure gradients on update steps
            if is_update_step:
                self.compute_amortized_gradients(inputs)

            self.model.zero_grad()
            self.reset_effective_weight_grads()

            with torch.cuda.amp.autocast():
                student_evidence = self.model(inputs)
                
                # Knowledge Distillation from Teacher Dirichlet Distribution
                distill_loss = torch.tensor(0.0, device=device)
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_evidence = self.teacher_model(inputs)
                    
                    # Convert to alpha parameters for Dirichlet calculation
                    student_alpha = student_evidence + 1.0
                    teacher_alpha = teacher_evidence + 1.0
                    distill_loss = dirichlet_kl_divergence(student_alpha, teacher_alpha)

            # Evidential Focal Loss calculations are computed in FP32 to prevent lgamma underflow
            with torch.cuda.amp.autocast(enabled=False):
                classification_loss = self.criterion(student_evidence.float(), targets, epoch)
                
            # Composite Loss incorporating the scheduled distillation parameter
            loss = classification_loss + alpha_d * distill_loss
            scaled_loss = loss * current_loss_scale
            
            self.scaler.scale(scaled_loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            params_to_clip = [p for group in self.optimizer.param_groups for p in group['params']]
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            
            if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                if ema_grad is None:
                    ema_grad = grad_norm.item()
                else:
                    ema_grad = 0.95 * ema_grad + 0.05 * grad_norm.item()

            # Cache task loss weight gradients on update steps
            if is_update_step:
                inv_scale = 1.0 / (self.scaler.get_scale() + 1e-8)
                for m in self.model.modules():
                    if isinstance(m, (MDEPLinear, MDEPConv2d)):
                        if hasattr(m, 'effective_weight') and m.effective_weight.grad is not None:
                            m.grad_L_w = m.effective_weight.grad.clone().detach() * inv_scale
                        else:
                            m.grad_L_w = torch.zeros_like(m.weight)

            if not is_warmup and batch_idx == 0:
                self.check_gradient_flow(epoch)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Trigger multi-agent topology scores updates on update steps
            if not is_warmup and is_update_step:
                self.last_flop_rate = update_scores_agents(self.model)

            self.model.zero_grad()
            self.reset_effective_weight_grads()

            loss_val = loss.item()
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * loss_val

            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - epoch_start
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (num_batches - batch_idx - 1)
                
                print(
                    f"    Batch [{batch_idx+1:>4}/{num_batches}] "
                    f"| Loss: {ema_loss:.4f} "
                    f"| distill: {distill_loss.item():.4f} (alpha_d={alpha_d:.3f}) "
                    f"| LR: {current_lr:.2e} "
                    f"| Flop: {self.last_flop_rate * 100:.3f}% "
                    f"| ETA: {eta/60:.1f}m",
                    flush=True
                )
                
        return ema_loss
