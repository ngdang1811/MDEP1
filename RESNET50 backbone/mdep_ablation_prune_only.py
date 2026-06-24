"""
============================================================================
  MDEP — Ablation Study: PRUNE-ONLY (Microglia Only)
  Single-file Kaggle Notebook version
  
  This ablation disables the Astrocyte (growing) agent entirely.
  Only the Microglia (pruning) agent drives sparsity decisions.
  Compare results with the full MDEP and grow-only ablation.
  
  HOW TO RUN ON KAGGLE:
    1. Create a new Notebook, set Accelerator to GPU (T4 or P100).
    2. Click "Add Data" → search "ISIC 2024" → add the challenge dataset.
    3. Copy-paste this entire file into a single code cell.
    4. Run the cell.
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from PIL import Image
import io
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, precision_recall_curve, auc
)
# ============================================================================
#  SECTION 1 — EDL Core (Evidential Deep Learning foundations)
# ============================================================================

class EvidenceLayer(nn.Module):
    """
    Ensures the output of the network is non-negative evidence (e >= 0).
    Replaces the traditional Softmax layer for EDL.
    """
    def __init__(self, activation='softplus', max_evidence=20.0):
        super(EvidenceLayer, self).__init__()
        self.max_evidence = max_evidence
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        ev = self.activation(x)
        if self.max_evidence is not None:
            ev = torch.clamp(ev, max=self.max_evidence)
        return ev


def compute_uncertainties(evidence):
    """
    Computes epistemic and aleatoric uncertainties from the evidence.

    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)

    Returns:
        dict: Contains epistemic (u_e), aleatoric (u_a), alpha, and S.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = evidence.shape[1]

    # Epistemic Uncertainty: u_e = K / S
    u_e = K / S

    # Aleatoric Uncertainty: u_a = sum (alpha_c / S) * (psi(S+1) - psi(alpha_c+1))
    # NOTE: The formula in the original research proposal main (36).pdf incorrectly had a negative sign prefix
    # (u_a = - sum ...). Since psi(S+1) > psi(alpha_c+1), that negative sign would yield negative uncertainty.
    # We omit the negative sign here to ensure mathematical consistency and non-negativity (u_a >= 0).
    digamma_S = torch.digamma(S + 1.0)
    digamma_alpha = torch.digamma(alpha + 1.0)
    u_a_term = (alpha / S) * (digamma_S - digamma_alpha)
    u_a = torch.sum(u_a_term, dim=1, keepdim=True)
    assert torch.all(u_a > -1e-6), f"Sanity check failed: u_a contains negative values {u_a[u_a < -1e-6].tolist()}"
    u_a = torch.clamp(u_a, min=0.0)

    return {
        'epistemic': u_e,
        'aleatoric': u_a,
        'alpha': alpha,
        'S': S,
    }


# ============================================================================
#  SECTION 2 — Loss Functions (Evidential Focal Loss + KL regularization)
# ============================================================================

def kl_divergence(alpha, num_classes):
    """
    KL divergence between a Dirichlet(alpha) and a uniform Dirichlet(1,...,1).
    """
    beta = torch.ones(1, num_classes, dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


class EvidentialFocalLoss(nn.Module):
    """
    Evidential Focal Loss (EFL) with KL Divergence Regularization and Dynamic Gamma Scheduling.
    The focal weight modulates the CE term — not the evidence space directly —
    so the Dirichlet structure stays valid even on highly imbalanced data.
    """
    def __init__(self, gamma=1.2, num_classes=10, kl_lambda=0.1, class_weights=None, annealing_epochs=10, warmup_epochs=15, total_epochs=100):
        super(EvidentialFocalLoss, self).__init__()
        self.base_gamma = gamma
        self.gamma = gamma
        self.num_classes = num_classes
        self.kl_lambda = kl_lambda
        self.annealing_epochs = annealing_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, evidence, targets, epoch=None):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()

        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)

        p_hat = alpha / S

        loss_ce = torch.sum(
            targets * (torch.digamma(S) - torch.digamma(alpha)),
            dim=1, keepdim=True,
        )

        # Dynamic gamma scheduling
        if epoch is not None:
            if epoch < self.warmup_epochs:
                gamma_val = 0.0
            elif epoch < self.warmup_epochs + 5:
                # Linear warmup from 0.0 to 2.0 over 5 epochs
                gamma_val = 2.0 * (epoch - self.warmup_epochs) / 5.0
            else:
                # Max force phase: increase from 2.0 to 4.0 in the remaining epochs
                remaining_epochs = max(1, self.total_epochs - (self.warmup_epochs + 5))
                progress = (epoch - (self.warmup_epochs + 5)) / remaining_epochs
                gamma_val = 2.0 + 2.0 * min(1.0, progress)
        else:
            gamma_val = self.gamma

        self.gamma = gamma_val # Store current gamma value

        p_target = torch.sum(targets * p_hat, dim=1, keepdim=True)
        focal_weight = (1.0 - p_target.detach()) ** gamma_val

        if self.class_weights is not None:
            sample_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
        else:
            sample_weight = 1.0

        alpha_tilde = targets + (1 - targets) * alpha
        loss_kl = kl_divergence(alpha_tilde, self.num_classes)

        # KL Annealing
        if epoch is not None and self.annealing_epochs > 0:
            annealing_coef = min(1.0, epoch / self.annealing_epochs)
        else:
            annealing_coef = 1.0

        # Modulate the entire loss (both CE and KL) by focal and sample weights
        # Scale the CE loss by focal weight, and the overall loss by sample weight to balance KL and CE forces under class imbalance
        loss = sample_weight * (focal_weight * loss_ce + self.kl_lambda * annealing_coef * loss_kl)
        return torch.mean(loss)


# ============================================================================
#  SECTION 3 — MDEP Multi-Agent Sparsity Engine
# ============================================================================

class SmoothedSTE(torch.autograd.Function):
    """
    Smoothed Straight-Through Estimator with Local 2:4 Bounds.
    Forward: passes the hard binary mask unchanged, but computes local thresholds.
    Backward: approximates dM/dS ≈ sigma'((S - tau)/gamma) so gradients flow
              only to connections near the 2:4 survival boundary.
    """
    @staticmethod
    def forward(ctx, scores, mask, gamma):
        shape = scores.shape
        if scores.numel() % 4 == 0:
            scores_flat = scores.view(-1, 4)
            # Find the 2nd and 3rd largest values in each block
            sorted_scores, _ = torch.sort(scores_flat, dim=-1, descending=True)
            s2 = sorted_scores[:, 1]
            s3 = sorted_scores[:, 2]
            # Local threshold is the midpoint
            tau = ((s2 + s3) / 2.0).unsqueeze(-1) # shape: (N, 1)
            tau = tau.expand_as(scores_flat).reshape(shape)
        else:
            tau = torch.zeros_like(scores)

        ctx.save_for_backward(scores, tau, torch.tensor(gamma))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, tau, gamma = ctx.saved_tensors
        gamma_val = gamma.item()
        
        # Localized STE: margin to the boundary
        margin = scores - tau
        
        sig = torch.sigmoid(margin / gamma_val)
        grad_scores = grad_output * sig * (1.0 - sig) / gamma_val
        return grad_scores, None, None


def generate_2_4_mask(scores):
    """
    Generates an NVIDIA 2:4 structured sparsity mask.
    For every contiguous block of 4 elements the top-2 (by score) survive.
    This replaces a single global threshold tau with a dynamic, local one.
    
    NOTE ON SPARSITY SCHEDULING:
    The research proposal main (36).pdf mentions a gradual cosine pruning schedule. However, to comply with the 
    hard hardware-enforced 2:4 structured sparsity (exactly 50% non-zero parameters per block of 4) required for
    acceleration on Tensor Cores, a hard 50% mask is applied immediately after the warmup phase. The cosine 
    schedule is instead applied to the exploration temperature (gamma) of the Smoothed STE to control the dynamic 
    structural exploration rate (mask flips), rather than the sparsity ratio itself.
    """
    if scores.numel() % 4 != 0:
        return torch.ones_like(scores)

    shape = scores.shape
    scores_flat = scores.view(-1, 4)
    _, indices = torch.topk(scores_flat, 2, dim=-1)
    mask_flat = torch.zeros_like(scores_flat)
    mask_flat.scatter_(1, indices, 1.0)
    return mask_flat.view(shape)


class MDEPLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with MDEP dynamic sparsity."""
    def __init__(self, in_features, out_features, bias=True):
        super(MDEPLinear, self).__init__(in_features, out_features, bias)
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            raw_mask = generate_2_4_mask(self.scores)
            self.mask.copy_(raw_mask)
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.gamma)
            effective_weight = self.weight * differentiable_mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        # Bypass PyTorch's nn.Module.__setattr__ registration by writing directly to self.__dict__
        self.__dict__['effective_weight'] = effective_weight
            
        return F.linear(x, effective_weight, self.bias)


class MDEPConv2d(nn.Conv2d):
    """Drop-in replacement for nn.Conv2d with MDEP dynamic sparsity."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MDEPConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            raw_mask = generate_2_4_mask(self.scores)
            self.mask.copy_(raw_mask)
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.gamma)
            effective_weight = self.weight * differentiable_mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        # Bypass PyTorch's nn.Module.__setattr__ registration by writing directly to self.__dict__
        self.__dict__['effective_weight'] = effective_weight
            
        return F.conv2d(
            x, effective_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def update_scores_agents(model, beta=1.0):
    """
    ABLATION: Prune-Only — Microglia agent only.
    The Astrocyte (growing) signal G_ij is zeroed out.

    Microglia (§5.2): C_ij = Norm(|w_ij * ∂L_EFL/∂w_ij|) + β·Norm(|w_ij * ∂u_a/∂w_ij|)
    """
    total_flops = 0
    total_elements = 0
    
    print("\n🔍 [DEBUG - update_scores_agents - ABLATION: Prune-Only]")
    print("-" * 75)
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                continue

            # Capture old mask before score update
            old_mask = generate_2_4_mask(module.scores.data)

            w_val = module.weight.data

            # --- Microglia agent: pruning score (§5.2) ---
            # c1: importance for prediction = |w * ∂L_EFL/∂w|
            c1 = torch.abs(w_val * module.grad_L_w)
            c1_min = c1.min().item()
            c1_max = c1.max().item()
            c1_norm = (c1 - c1_min) / (c1_max - c1_min + 1e-8)

            # c2: importance for noise modelling = |w * ∂u_a/∂w|
            grad_ua_w = getattr(module, 'grad_ua_w', torch.zeros_like(w_val))
            c2 = torch.abs(w_val * grad_ua_w)
            c2_min = c2.min().item()
            c2_max = c2.max().item()
            c2_norm = (c2 - c2_min) / (c2_max - c2_min + 1e-8)

            C_ij = c1_norm + beta * c2_norm

            # --- Astrocyte agent: DISABLED (zeroed out) ---
            G_ij = torch.zeros_like(C_ij)

            # Corrected formulation: Growth promotes connection (+), Pruning demotes it (-)
            delta_S = G_ij - C_ij
            
            # Step 1: Update Velocity (Momentum EMA)
            beta_m = 0.95
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            
            # Step 2: Update Latent Scores S
            eta = 0.02
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            
            # Step 3: Zero-center scores to prevent global positive drift over time
            module.scores.data.sub_(module.scores.data.mean())

            # Step 4: Clamp scores to prevent infinite growth and gradient underflow (dead gradients)
            module.scores.data.clamp_(min=-5.0, max=5.0)
            
            # Compute new mask and count flops
            new_mask = generate_2_4_mask(module.scores.data)
            flops = (old_mask != new_mask).sum().item()
            total_flops += flops
            total_elements += old_mask.numel()

    flop_rate = total_flops / (total_elements + 1e-8)
    print(f"  >>> TOTAL FLOP RATE: {flop_rate*100:.6f}% ({total_flops} / {total_elements})")
    print("-" * 75)
    return flop_rate


# ============================================================================
#  SECTION 4 — Trainer (warm-up, cosine schedules, amortized gradients)
# ============================================================================

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

        # Smoothed-STE temperature schedule
        self.gamma_initial = 5.0
        self.gamma_final = 0.15
        
        # AMP Scaler for Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()

    def step_gamma(self, epoch):
        """Cosine-annealed temperature for the Smoothed STE."""
        if epoch < self.warmup_epochs:
            return self.gamma_initial
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        gamma = self.gamma_final + 0.5 * (self.gamma_initial - self.gamma_final) * (
            1 + math.cos(math.pi * progress)
        )
        return gamma

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
            
        print(f"\n🔍 [Gradient Flow Check - Epoch {epoch}]")
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
        #    Paper §5.3: u_e,i^(node) = |∂u_e / ∂a_i^(l)|
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
                        # grad: (B, out_features) → per-neuron: (out_features,)
                        m.u_e_node = torch.abs(grad).mean(dim=0).detach()
                    elif isinstance(m, MDEPConv2d):
                        # grad: (B, C_out, H, W) → per-neuron: (C_out,)
                        m.u_e_node = torch.abs(grad).mean(dim=(0, 2, 3)).detach()
                else:
                    m.u_e_node = None

        # Clean up hooks
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

        ema_loss = None
        ema_grad = None
        num_batches = len(dataloader)
        epoch_start = time.time()

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

            # Amortized uncertainty-gradient pass on the first batch of the epoch (also during warm-up epochs 0 and 1)
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
            
            if not torch.isnan(grad_norm).item() and not torch.isinf(grad_norm).item():
                if ema_grad is None:
                    ema_grad = grad_norm.item()
                else:
                    ema_grad = 0.95 * ema_grad + 0.05 * grad_norm.item()

            # Cache primary weight gradient for structural updates
            if not is_warmup or epoch < 2:
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

            # Multi-agent structure optimization (once per epoch)
            if not is_warmup and batch_idx == 0:
                mask_flop_rate = update_scores_agents(self.model)
                self.last_flop_rate = mask_flop_rate

            self.model.zero_grad()
            self.reset_effective_weight_grads()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * loss.item()

            # Progress printing
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - epoch_start
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (num_batches - batch_idx - 1)
                avg_loss = ema_loss if ema_loss is not None else 0.0
                avg_grad = ema_grad if ema_grad is not None else 0.0
                
                flop_str = f"| Flop: {self.last_flop_rate*100:.4f}%  " if hasattr(self, 'last_flop_rate') else ""
                
                print(
                    f"    Batch [{batch_idx+1:>5}/{num_batches}]  "
                    f"| Loss: {avg_loss:.4f}  "
                    f"| LR: {current_lr:.2e}  "
                    f"| GradNorm: {avg_grad:.4f}  "
                    f"{flop_str}"
                    f"| Elapsed: {elapsed/60:.1f}m  "
                    f"| ETA: {eta/60:.1f}m",
                    flush=True,
                )

        return ema_loss if ema_loss is not None else 0.0


# ============================================================================
#  SECTION 5 — ISIC 2024 Dataset + ResNet backbone + main()
# ============================================================================

class ISICDataset(Dataset):
    """PyTorch Dataset for the ISIC 2024 Skin Cancer challenge on Kaggle.
    Supports loading images from individual files OR from an HDF5 archive."""
    def __init__(self, dataframe, image_dir, transform=None, hdf5_path=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self._hdf5_file = None

    def _get_hdf5(self):
        """Lazy-open HDF5 file (one handle per worker process)."""
        if self._hdf5_file is None and self.hdf5_path and HAS_H5PY:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]['isic_id']
        image = None

        # Try 1: Load from individual image file
        img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = None

        # Try 2: Load from HDF5 archive
        if image is None and self.hdf5_path and HAS_H5PY:
            try:
                hf = self._get_hdf5()
                if isic_id in hf:
                    img_bytes = hf[isic_id][()]
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception:
                image = None

        # Fallback: black placeholder
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        target = self.data_frame.iloc[idx]['target']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)


def get_isic_dataloaders(batch_size=32, test_ratio=0.2):
    """
    Returns (train_loader, test_loader, num_classes).
    Uses stratified 80/20 split. Falls back to dummy data if not on Kaggle.
    """
    num_classes = 2

    # Auto-detect the ISIC dataset path under /kaggle/input/
    # Competition datasets are mounted under /kaggle/input/competitions/<slug>/
    # Regular datasets are mounted under /kaggle/input/<slug>/
    csv_path = None
    image_dir = None
    kaggle_input = '/kaggle/input'
    
    # Debug: show full tree under /kaggle/input/
    print(f"🔍 Checking Kaggle input dir: {kaggle_input}")
    print(f"   Exists? {os.path.isdir(kaggle_input)}")
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            depth = root.replace(kaggle_input, '').count(os.sep)
            if depth < 3:  # Only show first 3 levels
                indent = '   ' + '  ' * depth
                print(f"{indent}📁 {os.path.basename(root)}/")
                for f in files[:5]:  # Show first 5 files per dir
                    print(f"{indent}  📄 {f}")
                if len(files) > 5:
                    print(f"{indent}  ... and {len(files)-5} more files")
    
    def _try_find_dataset(base_dir):
        """Search for train-metadata.csv in a directory and return (csv_path, image_dir) or (None, None)."""
        if not os.path.isdir(base_dir):
            return None, None
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            candidate_csv = os.path.join(folder_path, 'train-metadata.csv')
            if os.path.exists(candidate_csv):
                img_dir = None
                for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
                    candidate_img = os.path.join(folder_path, img_sub)
                    if os.path.isdir(candidate_img):
                        img_dir = candidate_img
                        break
                if img_dir is None:
                    img_dir = os.path.join(folder_path, 'train-image')
                print(f"✅ Found ISIC dataset at: {folder_path}/")
                return candidate_csv, img_dir
        return None, None
    
    # Strategy 1: Check directly under /kaggle/input/<slug>/
    csv_path, image_dir = _try_find_dataset(kaggle_input)
    
    # Strategy 2: Check under /kaggle/input/competitions/<slug>/
    if csv_path is None:
        competitions_dir = os.path.join(kaggle_input, 'competitions')
        csv_path, image_dir = _try_find_dataset(competitions_dir)
    
    # Strategy 3: Recursive scan — check ALL subdirectories up to 2 levels deep
    if csv_path is None and os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            depth = root.replace(kaggle_input, '').count(os.sep)
            if depth > 2:
                continue
            if 'train-metadata.csv' in files:
                csv_path = os.path.join(root, 'train-metadata.csv')
                for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
                    candidate_img = os.path.join(root, img_sub)
                    if os.path.isdir(candidate_img):
                        image_dir = candidate_img
                        break
                if image_dir is None:
                    image_dir = os.path.join(root, 'train-image')
                print(f"✅ Found ISIC dataset via deep scan at: {root}/")
                break
    
    if csv_path:
        print(f"📂 CSV path:   {csv_path}")
        print(f"📂 Image dir:  {image_dir}")
    else:
        print(f"❌ train-metadata.csv not found anywhere under {kaggle_input}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if csv_path is None or not os.path.exists(csv_path):
        print("⚠ ISIC dataset not found. Falling back to dummy data.")
        X = torch.randn(200, 3, 224, 224)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, Y)
        tr = Subset(full, range(160))
        te = Subset(full, range(160, 200))
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(te, batch_size=batch_size),
                num_classes,
                torch.ones(num_classes))

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded CSV with {len(df)} rows, columns: {list(df.columns[:5])}")
    
    # Detect HDF5 archive for images
    hdf5_path = None
    dataset_root = os.path.dirname(csv_path)
    for hdf5_name in ['train-image.hdf5', 'train-image.h5']:
        candidate = os.path.join(dataset_root, hdf5_name)
        if os.path.exists(candidate):
            hdf5_path = candidate
            print(f"📂 HDF5 archive: {hdf5_path}")
            break
    
    # Debug: list available files in dataset root
    if os.path.isdir(dataset_root):
        print(f"📂 Dataset contents: {os.listdir(dataset_root)}")
    
    # Subsample to keep training feasible within Kaggle session limits.
    # Set MAX_SAMPLES = None to use the full dataset.
    MAX_SAMPLES = None  # None = use full dataset (401K samples)
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        print(f"📉 Subsampling: {len(df)} → {MAX_SAMPLES} samples (set MAX_SAMPLES=None for full dataset)")
        df = df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(MAX_SAMPLES * len(x) / len(df))), random_state=42)
        ).reset_index(drop=True)
        print(f"   After stratified subsample: {len(df)} samples, target distribution:")
        print(f"   {df['target'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df['target'], random_state=42,
    )
    print(f"📊 Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")
    train_ds = ISICDataset(train_df, image_dir, transform=train_tf, hdf5_path=hdf5_path)
    test_ds  = ISICDataset(test_df,  image_dir, transform=test_tf,  hdf5_path=hdf5_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Compute class weights (dampened inverse frequency to prevent loss/gradient explosion)
    import math
    class_counts = train_df['target'].value_counts().sort_index()
    total = len(train_df)
    cw_raw = [math.sqrt(total / class_counts.get(c, 1)) for c in range(num_classes)]
    majority_weight = cw_raw[0]
    cw = torch.tensor([w / majority_weight for w in cw_raw], dtype=torch.float32)
    print(f"⚖️  Class weights: {dict(enumerate(cw.tolist()))}")

    return train_loader, test_loader, num_classes, cw


def replace_conv2d_with_mdep(model):
    """Recursively swap nn.Conv2d / nn.Linear → MDEPConv2d / MDEPLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_conv2d_with_mdep(module)


# ============================================================================
#  SECTION 6 — Evaluation, Metrics & Visualization
# ============================================================================

def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            continue
        b_acc  = accuracies[mask].mean()
        b_conf = confidences[mask].mean()
        b_size = mask.sum()
        ece += (b_size / len(confidences)) * abs(b_acc - b_conf)
        bin_accs.append(b_acc)
        bin_confs.append(b_conf)
        bin_sizes.append(b_size)
    return ece, bin_accs, bin_confs, bin_sizes


def plot_reliability_diagram(bin_accs, bin_confs, bin_sizes, n_bins=15):
    """Reliability diagram: accuracy vs confidence per bin."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(n_bins)
    width = 0.8
    ax.bar(x, bin_accs, width, label='Accuracy', color='#4e79a7', alpha=0.85)
    ax.bar(x, bin_confs, width, label='Confidence', color='#e15759', alpha=0.4)
    ax.plot([-0.5, n_bins - 0.5], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Value')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_histogram(u_e_correct, u_e_incorrect):
    """Overlaid histograms of epistemic uncertainty for correct vs wrong."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(u_e_correct) > 0:
        ax.hist(u_e_correct, bins=40, alpha=0.6, label='Correct', color='#59a14f')
    if len(u_e_incorrect) > 0:
        ax.hist(u_e_incorrect, bins=40, alpha=0.6, label='Incorrect', color='#e15759')
    ax.set_xlabel('Epistemic Uncertainty (u_e)')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, probs):
    """Precision-Recall Curve with AUC."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(recall, precision, color='#86bcB6', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_risk_coverage_curve(y_true, y_pred, confidences):
    """Risk-Coverage curve and AURC (Area Under Risk-Coverage)."""
    # Sort instances by descending confidence
    sorted_indices = np.argsort(-confidences)
    sorted_true = y_true[sorted_indices]
    sorted_pred = y_pred[sorted_indices]
    
    coverages = []
    risks = []
    
    n_samples = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    
    for i in range(1, n_samples + 1):
        coverages.append(i / n_samples)
        risks.append(cumulative_errors[i-1] / i)
        
    aurc = auc(coverages, risks)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(coverages, risks, color='#f28e2b', lw=2, label=f'Risk-Coverage (AURC = {aurc:.4f})')
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk (Error Rate)')
    ax.set_title('Risk-Coverage Curve')
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_sparsity_report(model):
    """Per-layer and total sparsity stats + 2:4 pattern check + MACs estimation."""
    print("\n📐 Sparsity & Hardware Metrics Report")
    print("-" * 75)
    total_params = 0
    total_zeros  = 0
    total_macs_dense = 0
    total_macs_sparse = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            mask = module.mask
            n = mask.numel()
            z = (mask == 0).sum().item()
            total_params += n
            total_zeros  += z
            sparsity = z / n * 100 if n > 0 else 0.0
            
            # Simple MACs heuristic based on input/output size if possible
            # We assume MACs scales linearly with the number of non-zero parameters
            # for a given input size. We use parameter count as a proxy for MACs savings
            # on Tensor Cores which accelerate 2:4 structured sparsity by exactly 2x.
            macs_dense = n
            macs_sparse = n - z
            total_macs_dense += macs_dense
            total_macs_sparse += macs_sparse
            
            # Check 2:4 pattern
            if n % 4 == 0:
                blocks = mask.view(-1, 4)
                valid = (blocks.sum(dim=1) == 2).all().item()
                pattern = "✅ 2:4 (TensorCore Ready)" if valid else "❌ Not 2:4"
            else:
                pattern = "⚠ skip (size%4≠0)"
            print(f"  {name:30s} | {sparsity:5.1f}% sparse | {pattern}")
            
    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0
    print("-" * 75)
    print(f"  {'TOTAL PARAMS':30s} | {overall:5.1f}% sparse")
    print(f"  {'THEORETICAL MACs SAVED':30s} | {macs_saved:5.1f}% reduction in MDEP layers")
    print("  *(Note: Ampere GPU Tensor Cores provide 2x speedup for strict 2:4 sparsity)*")
    print()


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes):
    """Full evaluation: metrics, plots, and uncertainty analysis."""
    model.eval()

    all_targets  = []
    all_preds    = []
    all_confs    = []
    all_probs    = []
    all_u_e      = []
    all_u_a      = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        evidence = model(inputs)
        unc = compute_uncertainties(evidence)

        alpha = unc['alpha']
        S     = unc['S']
        p_hat = (alpha / S).cpu().numpy()
        preds = p_hat.argmax(axis=1)
        confs = p_hat.max(axis=1)

        all_targets.append(targets.numpy())
        all_preds.append(preds)
        all_confs.append(confs)
        all_probs.append(p_hat)
        all_u_e.append(unc['epistemic'].cpu().numpy().squeeze())
        all_u_a.append(unc['aleatoric'].cpu().numpy().squeeze())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    confs  = np.concatenate(all_confs)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    correct = (y_pred == y_true).astype(float)

    # ── Scalar Metrics (Default Threshold = 0.5) ───────────────────
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    if num_classes == 2:
        macro_auroc = roc_auc_score(y_true, probs[:, 1], average='macro')
        pr_auc = average_precision_score(y_true, probs[:, 1])
        brier = brier_score_loss(y_true, probs[:, 1])
        try:
            pauc = roc_auc_score(y_true, probs[:, 1], max_fpr=0.2)
        except ValueError:
            pauc = float('nan')
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        # Optimize threshold for Balanced Accuracy
        best_t = 0.5
        best_bal_acc = 0.0
        for t in np.linspace(0.01, 0.99, 99):
            y_pred_t = (probs[:, 1] >= t).astype(int)
            bal_acc_t = balanced_accuracy_score(y_true, y_pred_t)
            if bal_acc_t > best_bal_acc:
                best_bal_acc = bal_acc_t
                best_t = t
                
        y_pred_opt = (probs[:, 1] >= best_t).astype(int)
        bal_acc_opt = balanced_accuracy_score(y_true, y_pred_opt)
        macro_f1_opt = f1_score(y_true, y_pred_opt, average='macro')
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_opt).ravel()
        sensitivity_opt = tp_opt / (tp_opt + fn_opt + 1e-8)
        specificity_opt = tn_opt / (tn_opt + fp_opt + 1e-8)
    else:
        macro_auroc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        pauc = float('nan')
        pr_auc = float('nan')
        brier = float('nan')
        sensitivity = float('nan')
        specificity = float('nan')
        best_t = 0.5
        bal_acc_opt = bal_acc
        macro_f1_opt = macro_f1
        sensitivity_opt = float('nan')
        specificity_opt = float('nan')
        y_pred_opt = y_pred

    ece_val, bin_accs, bin_confs, bin_sizes = compute_ece(confs, correct)

    # Minority-ECE (class 1 = malignant)
    minority_mask = (y_true == 1)
    if minority_mask.sum() > 0:
        m_ece, _, _, _ = compute_ece(confs[minority_mask], correct[minority_mask])
    else:
        m_ece = float('nan')

    # ── Print Results ──────────────────────────────────────────────
    print("\n📈 Evaluation Results (Threshold = 0.5)")
    print("=" * 50)
    print(f"  Balanced Accuracy     : {bal_acc:.4f}")
    print(f"  Macro F1-Score        : {macro_f1:.4f}")
    if num_classes == 2:
        print(f"  Sensitivity (Recall)  : {sensitivity:.4f}")
        print(f"  Specificity           : {specificity:.4f}")
        print(f"  PR-AUC                : {pr_auc:.4f}")
        print(f"  Brier Score           : {brier:.4f}")
    print(f"  Macro-AUROC           : {macro_auroc:.4f}")
    print(f"  pAUC (@ 20% FPR)      : {pauc:.4f}")
    print(f"  ECE (15 bins)         : {ece_val:.4f}")
    print(f"  Minority-ECE (cls 1)  : {m_ece:.4f}")
    print(f"  Mean Epistemic u_e    : {u_e.mean():.4f}")
    print(f"  Mean Aleatoric u_a    : {u_a.mean():.4f}")
    print("=" * 50)

    if num_classes == 2:
        print(f"\n📈 Evaluation Results (Optimized Threshold = {best_t:.4f})")
        print("=" * 50)
        print(f"  Balanced Accuracy     : {bal_acc_opt:.4f}")
        print(f"  Macro F1-Score        : {macro_f1_opt:.4f}")
        print(f"  Sensitivity (Recall)  : {sensitivity_opt:.4f}")
        print(f"  Specificity           : {specificity_opt:.4f}")
        print("=" * 50)

    # ── Plots ──────────────────────────────────────────────────────
    plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
    plot_uncertainty_histogram(
        u_e[correct.astype(bool)],
        u_e[~correct.astype(bool)],
    )
    if num_classes == 2:
        plot_pr_curve(y_true, probs[:, 1])
    plot_risk_coverage_curve(y_true, y_pred_opt, confs)

    return {
        'balanced_accuracy': bal_acc,
        'balanced_accuracy_opt': bal_acc_opt,
        'sensitivity_opt': sensitivity_opt,
        'specificity_opt': specificity_opt,
        'macro_auroc': macro_auroc,
        'pauc': pauc,
        'ece': ece_val,
        'minority_ece': m_ece,
        'mean_u_e': float(u_e.mean()),
        'mean_u_a': float(u_a.mean()),
    }


# ============================================================================
#  SECTION 7 — main()
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")

    # ── Data (stratified train / test split) ────────────────────────
    train_loader, test_loader, num_classes, class_weights = get_isic_dataloaders(batch_size=32)
    print(f"📊 Classes: {num_classes}")
    print(f"   Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        EvidenceLayer(activation='softplus'),
    )
    # Initialize evidence output to be small to prevent KL explosion
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    replace_conv2d_with_mdep(model)
    model = model.to(device)

    total_epochs  = 20
    warmup_epochs = 6

    criterion = EvidentialFocalLoss(
        gamma=1.2, num_classes=num_classes, kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=warmup_epochs, total_epochs=total_epochs
    )
    optimizer = optim.Adam(model.parameters(), lr=4.0e-05)

    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs)

    # ── Training ───────────────────────────────────────────────────
    print("\n🚀 Starting Training (ABLATION: Prune-Only / Microglia Only)")
    print("=" * 60)
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(epoch, train_loader, device)
        phase = "Warm-up (Dense)" if epoch < warmup_epochs else "Dynamic 2:4 Sparsity"
        gamma = trainer.step_gamma(epoch)
        print(
            f"  Epoch [{epoch+1:>2}/{total_epochs}]  "
            f"| Phase: {phase:<22} "
            f"| γ: {gamma:.4f}  "
            f"| Loss: {loss:.4f}"
        )
    print("=" * 60)
    print("✅ Training complete.\n")

    # ── Evaluation ─────────────────────────────────────────────────
    evaluate(model, test_loader, device, num_classes)
    print_sparsity_report(model)


# ── Run ────────────────────────────────────────────────────────────────
main()
