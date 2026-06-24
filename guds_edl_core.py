"""
============================================================================
  MDEP — Microglial-Driven Evidential Pruning
  Single-file Kaggle Notebook version
  
  HOW TO RUN ON KAGGLE:
    1. Create a new Notebook, set Accelerator to GPU (T4 or P100).
    2. Click "Add Data" → search "ISIC 2024" → add the challenge dataset.
    3. Copy-paste this entire file into a single code cell.
    4. Run the cell.
============================================================================
"""

import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
import matplotlib
import sys
if not hasattr(sys, 'ps1') and 'IPython' not in sys.modules:
    matplotlib.use('Agg')
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
#  SECTION 1 — EDL Core (Evidential Deep Learning foundations)
# ============================================================================

def sinkhorn_knopp(logits, n_iters=10, eps=1e-6):
    """
    Sinkhorn-Knopp algorithm to normalize a square matrix into a doubly stochastic matrix.
    logits: (N, N)
    """
    K = torch.exp(logits - torch.max(logits, dim=-1, keepdim=True)[0])
    for _ in range(n_iters):
        # Normalize rows
        K = K / (K.sum(dim=1, keepdim=True) + eps)
        # Normalize columns
        K = K / (K.sum(dim=0, keepdim=True) + eps)
    return K


def permutation_penalty(M):
    """
    Lipschitz-continuous l1-l2 penalty from AutoShuffleNet.
    For doubly stochastic matrix M, row sums and col sums are 1,
    so l1-norm is 1, and the penalty is:
    P(M) = sum_i (1 - ||M_{i:}||_2) + sum_j (1 - ||M_{:j}||_2)
    """
    N = M.shape[0]
    row_norms = torch.norm(M, p=2, dim=1)
    col_norms = torch.norm(M, p=2, dim=0)
    penalty = 2.0 * N - row_norms.sum() - col_norms.sum()
    return penalty


def get_permutation_loss(model, penalty_weight=0.01):
    """
    Aggregate permutation penalty from all active (unfrozen) permutation layers.
    """
    total_penalty = 0.0
    count = 0
    for module in model.modules():
        if hasattr(module, 'freeze_perm') and hasattr(module, 'get_doubly_stochastic_matrix'):
            if module.freeze_perm[0] == 0:
                M = module.get_doubly_stochastic_matrix()
                total_penalty = total_penalty + permutation_penalty(M)
                count += 1
    if count > 0:
        return total_penalty * penalty_weight
    # Get device dynamically
    device = next(model.parameters()).device
    return torch.tensor(0.0, device=device)


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
        self.register_buffer('logit_adjustment', torch.zeros(1))

    def forward(self, x):
        if self.logit_adjustment.shape == x.shape[-1:] or self.logit_adjustment.numel() == x.shape[-1]:
            x = x + self.logit_adjustment
        ev = self.activation(x)
        if self.max_evidence is not None:
            ev = torch.clamp(ev, max=self.max_evidence)
        return ev


def compute_uncertainties(evidence, alpha_prior=None):
    """
    Computes epistemic and aleatoric uncertainties from the evidence.

    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)
        alpha_prior (torch.Tensor or float, optional): Dirichlet prior parameters. Default is ones.

    Returns:
        dict: Contains epistemic (u_e), aleatoric (u_a), alpha, and S.
    """
    if alpha_prior is None:
        alpha_prior = torch.ones(evidence.shape[1], device=evidence.device)
    elif not isinstance(alpha_prior, torch.Tensor):
        alpha_prior = torch.tensor(alpha_prior, dtype=torch.float32, device=evidence.device)
    else:
        alpha_prior = alpha_prior.to(evidence.device)

    alpha = evidence + alpha_prior
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = evidence.shape[1]

    # Epistemic Uncertainty: u_e = K / S
    u_e = K / S

    # Aleatoric Uncertainty: u_a = sum (alpha_c / S) * (psi(S+1) - psi(alpha_c+1))
    digamma_S = torch.digamma(S + 1.0)
    digamma_alpha = torch.digamma(alpha + 1.0)
    u_a_term = (alpha / S) * (digamma_S - digamma_alpha)
    u_a = torch.sum(u_a_term, dim=1, keepdim=True)
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
    def __init__(self, gamma=1.2, num_classes=10, kl_lambda=0.1, class_weights=None, annealing_epochs=10, warmup_epochs=15, total_epochs=100, disable_efl=False, kl_scaling='asymmetric'):
        super(EvidentialFocalLoss, self).__init__()
        self.base_gamma = gamma
        self.gamma = gamma
        self.num_classes = num_classes
        self.kl_lambda = kl_lambda
        self.annealing_epochs = annealing_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.disable_efl = disable_efl
        self.kl_scaling = kl_scaling
        # class_weights: tensor of shape (num_classes,) — higher weight for rare classes
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, evidence, targets, epoch=None):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()

        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)

        # Expected probability
        p_hat = alpha / S

        # Cross-entropy term: sum_c  y_c * (psi(S) - psi(alpha_c))
        loss_ce = torch.sum(
            targets * (torch.digamma(S) - torch.digamma(alpha)),
            dim=1, keepdim=True,
        )

        # Dynamic gamma scheduling (Cosine Ramp Schedule as per Phase 4)
        if epoch is not None and not self.disable_efl:
            if epoch < self.warmup_epochs:
                gamma_val = 0.0
            else:
                import math
                progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                gamma_val = self.base_gamma * 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            gamma_val = self.gamma if not self.disable_efl else 0.0

        self.gamma = gamma_val # Store current gamma value

        # Focal modulation on the true-class probability
        p_target = torch.sum(targets * p_hat, dim=1, keepdim=True)
        if self.disable_efl:
            focal_weight = torch.ones_like(p_target)
        else:
            focal_weight = (1.0 - p_target.detach()) ** gamma_val

        # Per-sample class weight: look up weight for the true class
        if self.class_weights is not None:
            sample_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
        else:
            sample_weight = 1.0

        # KL regularization — shrink evidence for *incorrect* classes toward 0
        alpha_tilde = targets + (1 - targets) * alpha
        base_loss_kl = kl_divergence(alpha_tilde, self.num_classes)
        
        # Bounded Asymmetric Scaling (Phase 4 & 5 Math update)
        if self.kl_scaling == 'asymmetric' and self.class_weights is not None:
            # Lambda_asym = min(omega_y, 10.0)
            true_class_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
            Lambda_asym = torch.clamp(true_class_weight, max=10.0)
        else:
            Lambda_asym = 1.0

        # KL Annealing
        if epoch is not None and self.annealing_epochs > 0:
            annealing_coef = min(1.0, epoch / self.annealing_epochs)
        else:
            annealing_coef = 1.0

        loss = sample_weight * focal_weight * (loss_ce + self.kl_lambda * annealing_coef * Lambda_asym * base_loss_kl)
        return torch.mean(loss)


# ============================================================================
#  SECTION 3 — MDEP Multi-Agent Sparsity Engine
# ============================================================================


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
    """Drop-in replacement for nn.Linear with MDEP dynamic sparsity and learned permutation."""
    def __init__(self, in_features, out_features, bias=True):
        super(MDEPLinear, self).__init__(in_features, out_features, bias)
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0
        self.warmup = True
        
        # PA-DST permutation parameters
        self.perm_logits = nn.Parameter(torch.eye(in_features) * 5.0 + torch.randn(in_features, in_features) * 0.01)
        self.register_buffer('freeze_perm', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('perm_indices', torch.arange(in_features, dtype=torch.long))

    def get_doubly_stochastic_matrix(self):
        return sinkhorn_knopp(self.perm_logits)

    @torch.no_grad()
    def freeze_permutation(self):
        M = self.get_doubly_stochastic_matrix()
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-M.cpu().numpy())
            perm_indices = torch.tensor(col_ind, dtype=torch.long, device=self.weight.device)
        except ImportError:
            M_cpu = M.detach().cpu()
            perm_indices = torch.zeros(M_cpu.shape[0], dtype=torch.long)
            used = set()
            for i in range(M_cpu.shape[0]):
                sorted_idx = torch.argsort(M_cpu[i], descending=True)
                for idx in sorted_idx:
                    if idx.item() not in used:
                        perm_indices[i] = idx
                        used.add(idx.item())
                        break
            perm_indices = perm_indices.to(self.weight.device)
        self.perm_indices.copy_(perm_indices)
        self.freeze_perm[0] = 1

    def forward(self, x):
        # Apply permutation
        if self.freeze_perm[0] == 1:
            x = x.index_select(-1, self.perm_indices)
        else:
            M = self.get_doubly_stochastic_matrix()
            x = torch.matmul(x, M)

        if self.warmup:
            effective_weight = self.weight
        else:
            raw_mask = generate_2_4_mask(self.scores)
            self.mask.copy_(raw_mask)
            effective_weight = self.weight * self.mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        # Bypass PyTorch's nn.Module.__setattr__ registration by writing directly to self.__dict__
        self.__dict__['effective_weight'] = effective_weight
            
        return F.linear(x, effective_weight, self.bias)


class MDEPConv2d(nn.Conv2d):
    """Drop-in replacement for nn.Conv2d with MDEP dynamic sparsity and learned permutation."""
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
        
        # PA-DST permutation parameters
        self.perm_logits = nn.Parameter(torch.eye(in_channels) * 5.0 + torch.randn(in_channels, in_channels) * 0.01)
        self.register_buffer('freeze_perm', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('perm_indices', torch.arange(in_channels, dtype=torch.long))

    def get_doubly_stochastic_matrix(self):
        return sinkhorn_knopp(self.perm_logits)

    @torch.no_grad()
    def freeze_permutation(self):
        M = self.get_doubly_stochastic_matrix()
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-M.cpu().numpy())
            perm_indices = torch.tensor(col_ind, dtype=torch.long, device=self.weight.device)
        except ImportError:
            M_cpu = M.detach().cpu()
            perm_indices = torch.zeros(M_cpu.shape[0], dtype=torch.long)
            used = set()
            for i in range(M_cpu.shape[0]):
                sorted_idx = torch.argsort(M_cpu[i], descending=True)
                for idx in sorted_idx:
                    if idx.item() not in used:
                        perm_indices[i] = idx
                        used.add(idx.item())
                        break
            perm_indices = perm_indices.to(self.weight.device)
        self.perm_indices.copy_(perm_indices)
        self.freeze_perm[0] = 1

    def forward(self, x):
        # Apply permutation
        if self.freeze_perm[0] == 1:
            x = x.index_select(1, self.perm_indices)
        else:
            M = self.get_doubly_stochastic_matrix()
            x = torch.einsum('bchw,cd->bdhw', x, M)

        if self.warmup:
            effective_weight = self.weight
        else:
            raw_mask = generate_2_4_mask(self.scores)
            self.mask.copy_(raw_mask)
            effective_weight = self.weight * self.mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        # Bypass PyTorch's nn.Module.__setattr__ registration by writing directly to self.__dict__
        self.__dict__['effective_weight'] = effective_weight
            
        return F.conv2d(
            x, effective_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def update_scores_agents(model, beta=1.0, epoch=None, disable_pruner=False, disable_regrower=False, pruner_type='signed_first_order', use_anticryst=True, verbose=False):
    """
    Updates latent scores S_ij using Microglia (pruning) and Astrocyte (growing) signals.
    Also computes and returns the Mask Flop Rate (structural convergence) and Dead Channel Ratio.
    Saves a layer-wise histogram of delta_S if epoch is provided.
    """
    total_flops = 0
    total_elements = 0
    total_dead_channels = 0
    total_channels = 0
    delta_s_dict = {}
    
    if verbose:
        print("\n🔍 [DEBUG - update_scores_agents]")
        print("-" * 75)
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_microglia_w'):
                if verbose:
                    print(f"  Layer {name}: ⚠️ Missing grad_microglia_w")
                continue

            # Capture old mask before score update
            old_mask = generate_2_4_mask(module.scores.data)

            w_val = module.weight.data

            # --- Microglia agent: pruning score (§5.2) ---
            if disable_pruner:
                C_ij = torch.zeros_like(module.scores.data)
                c1_min, c1_max = 0.0, 0.0
            else:
                if pruner_type == 'signed_first_order':
                    c1 = torch.relu(w_val * module.grad_microglia_w)
                else: # 'absolute_grad' (old baseline)
                    c1 = torch.abs(w_val * module.grad_microglia_w)
                    
                c1_min = c1.min().item()
                c1_max = c1.max().item()
                if c1_max - c1_min > 1e-8:
                    c1_flat = c1.view(-1)
                    C_ij = (c1_flat.argsort().argsort().float() / (c1_flat.numel() - 1)).view_as(c1)
                else:
                    C_ij = torch.zeros_like(c1)

            # --- Astrocyte agent: growing score (§5.3) ---
            if disable_regrower:
                G_ij = torch.zeros_like(module.scores.data)
                g2_min, g2_max = 0.0, 0.0
            else:
                g2 = torch.abs(module.grad_astrocyte_w)
                g2_min = g2.min().item()
                g2_max = g2.max().item()
                if g2_max - g2_min > 1e-8:
                    g2_flat = g2.view(-1)
                    G_ij = (g2_flat.argsort().argsort().float() / (g2_flat.numel() - 1)).view_as(g2)
                else:
                    G_ij = torch.zeros_like(g2)

            # Khắc phục hiện tượng kết tinh Astrocyte (Phase 4 Action 2):
            g1_max = 0.0
            if use_anticryst and not disable_regrower:
                u_e_node = getattr(module, 'u_e_node', None)
                if u_e_node is not None:
                    if isinstance(module, MDEPLinear):
                        g1 = u_e_node.unsqueeze(1).expand_as(w_val)
                    elif isinstance(module, MDEPConv2d):
                        g1 = u_e_node.view(-1, 1, 1, 1).expand_as(w_val)
                    else:
                        g1 = torch.ones_like(w_val)
                else:
                    g1 = torch.ones_like(w_val)
                
                g1_min = g1.min().item()
                g1_max = g1.max().item()
                if g1_max - g1_min > 1e-8:
                    u_e_flat = u_e_node.view(-1)
                    u_e_rank = (u_e_flat.argsort().argsort().float() / (u_e_flat.numel() - 1)).view_as(u_e_node)
                    if isinstance(module, MDEPLinear):
                        g1_norm = u_e_rank.unsqueeze(1).expand_as(w_val)
                    elif isinstance(module, MDEPConv2d):
                        g1_norm = u_e_rank.view(-1, 1, 1, 1).expand_as(w_val)
                    else:
                        g1_norm = torch.ones_like(w_val)
                else:
                    g1_norm = torch.zeros_like(g1)

                if G_ij.max().item() <= 1e-8:
                    noise = 0.0316 * torch.randn_like(G_ij) * g1_norm
                    G_ij = G_ij + torch.clamp(noise, min=0.0)

            # Calculate total driving force Delta S
            delta_S = G_ij - C_ij
            
            # Save for visualization
            if epoch is not None:
                delta_s_dict[name] = delta_S.detach().cpu().numpy()
            
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
            
            # Calculate dead channels (all weights pruned in a filter/row)
            if isinstance(module, MDEPLinear):
                dead = (new_mask.sum(dim=1) == 0).sum().item()
                c_count = new_mask.shape[0]
            elif isinstance(module, MDEPConv2d):
                dead = (new_mask.view(new_mask.shape[0], -1).sum(dim=1) == 0).sum().item()
                c_count = new_mask.shape[0]
            total_dead_channels += dead
            total_channels += c_count

            if verbose:
                print(f"  Layer: {name}")
                print(f"    grad_microglia : min={c1_min:.2e}, max={c1_max:.2e}")
                print(f"    grad_astrocyte : min={g2_min:.2e}, max={g2_max:.2e}")
                print(f"    u_e_node   : max={g1_max:.2e}")
                print(f"    delta_S    : min={delta_S.min().item():.4f}, max={delta_S.max().item():.4f}")
                print(f"    Flips/Total: {flops} / {old_mask.numel()} ({flops / old_mask.numel() * 100:.4f}%) | Dead Ch: {dead}/{c_count}")
                print("-" * 50)

    flop_rate = total_flops / (total_elements + 1e-8)
    dead_ratio = total_dead_channels / (total_channels + 1e-8)
    if verbose:
        print(f"  >>> TOTAL FLOP RATE: {flop_rate*100:.6f}% ({total_flops} / {total_elements})")
        print(f"  >>> TOTAL DEAD CHANNEL RATIO: {dead_ratio*100:.2f}% ({total_dead_channels} / {total_channels})")
    
    # Layer-wise Visualization of Delta S
    if verbose and epoch is not None and len(delta_s_dict) > 0:
        try:
            import os
            import math
            import matplotlib.pyplot as plt
            
            artifacts_dir = os.path.join(os.getcwd(), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            num_layers = len(delta_s_dict)
            cols = 4
            rows = math.ceil(num_layers / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
            
            # Handle the case where there is only one row or one cell
            if num_layers == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
                
            for idx, (name, dS) in enumerate(delta_s_dict.items()):
                ax = axes[idx]
                ax.hist(dS.flatten(), bins=50, color='purple', alpha=0.7)
                # Keep title short to fit
                short_name = name.split('.')[-1] if '.' in name else name
                if len(short_name) > 15:
                    short_name = short_name[:15] + ".."
                ax.set_title(short_name)
                ax.set_xlim(-1.1, 1.1)
            
            for idx in range(len(delta_s_dict), len(axes)):
                axes[idx].axis('off')
                
            plt.tight_layout()
            plot_path = os.path.join(artifacts_dir, f"delta_S_epoch_{epoch}.png")
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"  >>> Saved delta_S histogram to: {plot_path}")
        except Exception as e:
            print(f"  >>> Failed to save delta_S plot: {e}")

    if verbose:
        print("-" * 75)
    return flop_rate


# ============================================================================
#  SECTION 4 — Trainer (warm-up, cosine schedules, amortized gradients)
# ============================================================================

class MDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs=None, args=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.args = args
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
            grad_microglia = getattr(m, 'grad_microglia_w', None)
            grad_astrocyte = getattr(m, 'grad_astrocyte_w', None)
            
            if grad_microglia is None or grad_astrocyte is None:
                print(f"Layer {name}: grad_microglia_w or grad_astrocyte_w is None. Cannot perform flow check.")
                continue
                
            grad_microglia = grad_microglia.cpu()
            grad_astrocyte = grad_astrocyte.cpu()
            
            # Magnitudes in Taylor weight-space: |w * grad_w|
            mag_ua = torch.abs(w_val * grad_microglia)
            mag_L = torch.abs(grad_astrocyte)
            
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
            print(f"  |dL_Astrocyte/dw| (Raw): mean={mag_L.mean().item():.2e}, std={mag_L.std().item():.2e}, max={mag_L.max().item():.2e}")
            
            # Relative scale check (beta balance)
            ratio = mag_ua.mean() / (mag_L.mean() + 1e-8)
            print(f"  Ratio (Raw) Microglia / Astrocyte: {ratio.item():.4f}")
            
            # Plot histograms
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Flatten to 1D
            mag_ua_flat = mag_ua.numpy().flatten()
            mag_L_flat = mag_L.numpy().flatten()
            
            ax1.hist(mag_ua_flat, bins=50, color='blue', alpha=0.7)
            ax1.set_title(f'Microglia Score ({name})')
            ax1.set_xlabel('Magnitude')
            ax1.set_ylabel('Count')
            
            ax2.hist(mag_L_flat, bins=50, color='green', alpha=0.7)
            ax2.set_title(f'Astrocyte Score ({name})')
            ax2.set_xlabel('Magnitude')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            plot_path = os.path.join(artifacts_dir, f"grad_flow_epoch_{epoch}.png")
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
          • ∂(u_a/u_e) / ∂w   → signal for the Microglia agent (Relative Entropy Gradient)
          • ∂KL(Dir||1) / ∂w  → signal for the Astrocyte agent (Reverse KL Divergence)
          • ∂u_e / ∂a^(l)     → signal for the Astrocyte agent (per-neuron anti-crystallization)
        Called only once per epoch to keep FLOPs low.
        """
        self.model.train()

        # Register forward hooks to capture layer activations for Astrocyte anti-crystallization
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
        u_e = torch.mean(uncertainties['epistemic'])

        # 1. Microglia Agent (Relative Entropy Gradient)
        loss_microglia = torch.mean(uncertainties['aleatoric'] / (uncertainties['epistemic'] + 1e-6))
        self.model.zero_grad()
        self.reset_effective_weight_grads()
        loss_microglia.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if hasattr(m, 'effective_weight') and m.effective_weight.grad is not None:
                    m.grad_microglia_w = m.effective_weight.grad.clone().detach()
                else:
                    m.grad_microglia_w = torch.zeros_like(m.weight)

        # 2. Astrocyte Agent (Reverse KL Divergence Gradient)
        num_classes = outputs.shape[1]
        
        # Phase 5: Support class_conditioned vs uniform KL for Astrocyte
        regrower_type = getattr(self.args, 'regrower_type', 'kl_uniform') if self.args else 'kl_uniform'
        alpha_base = uncertainties['alpha']
        
        if regrower_type == 'class_conditioned' and getattr(self.criterion, 'class_weights', None) is not None:
            # We scale the alpha by class weights before computing KL, simulating focal expansion
            cw = self.criterion.class_weights.unsqueeze(0)
            loss_astrocyte = torch.mean(kl_divergence(alpha_base * cw, num_classes))
        else:
            loss_astrocyte = torch.mean(kl_divergence(alpha_base, num_classes))
            
        self.model.zero_grad()
        self.reset_effective_weight_grads()
        loss_astrocyte.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if hasattr(m, 'effective_weight') and m.effective_weight.grad is not None:
                    m.grad_astrocyte_w = m.effective_weight.grad.clone().detach()
                else:
                    m.grad_astrocyte_w = torch.zeros_like(m.weight)

        # Clear grads of all parameters and intermediate weight tensors to isolate the u_e graph
        self.model.zero_grad()
        self.reset_effective_weight_grads()

        # 3. ∂u_e/∂a^(l) → Astrocyte agent (per-neuron signal) for anti-crystallization
        #    Paper §5.3: u_e,i^(node) = |∂u_e / ∂a_i^(l)|
        act_tensors = []
        act_modules = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)) and name in activations:
                act_tensors.append(activations[name])
                act_modules.append(m)

        if act_tensors:
            grads = torch.autograd.grad(u_e, act_tensors, allow_unused=True)
            for m, grad in zip(act_modules, grads):
                if grad is not None:
                    if isinstance(m, MDEPLinear):
                        # grad: (B, ..., out_features) → per-neuron: (out_features,)
                        m.u_e_node = torch.abs(grad).mean(dim=tuple(range(grad.ndim - 1))).detach()
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

        # Freezing permutations when transition to sparsity happens
        if epoch == self.warmup_epochs:
            print("❄️ Freezing learned permutations into hard index maps...")
            for module in self.model.modules():
                if hasattr(module, 'freeze_permutation'):
                    module.freeze_permutation()

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

        disable_topology_cache = getattr(self.args, 'disable_topology_cache', False) if self.args else False

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

            # Amortized uncertainty-gradient pass. The default caches one structural
            # pass per epoch; the no-cache ablation recomputes it for each batch.
            structural_probe_batch = batch_idx == 0 or (disable_topology_cache and not is_warmup)
            if (not is_warmup or epoch < 2) and structural_probe_batch:
                self.compute_amortized_gradients(inputs)

            self.model.zero_grad()
            self.reset_effective_weight_grads()
            
            # Use Automatic Mixed Precision for Forward Pass
            with torch.amp.autocast('cuda'):
                evidence = self.model(inputs)
                
            # Ensure Evidential Loss runs strictly in FP32 to avoid digamma/log underflow
            with torch.amp.autocast('cuda', enabled=False):
                loss = self.criterion(evidence.float(), targets, epoch)
                if is_warmup:
                    perm_loss = get_permutation_loss(self.model, penalty_weight=0.01)
                    loss = loss + perm_loss
            
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

            verbose_structural_logs = getattr(self.args, 'verbose_structural_logs', False) if self.args else False

            if not is_warmup and batch_idx == 0 and verbose_structural_logs:
                self.check_gradient_flow(epoch)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Multi-agent structure optimization (once per epoch)
            if not is_warmup and structural_probe_batch:
                disable_pruner = getattr(self.args, 'disable_pruner', False) if self.args else False
                disable_regrower = getattr(self.args, 'disable_regrower', False) if self.args else False
                pruner_type = getattr(self.args, 'pruner_type', 'signed_first_order') if self.args else 'signed_first_order'
                use_anticryst = getattr(self.args, 'use_anticryst', True) if self.args else True
                
                mask_flop_rate = update_scores_agents(
                    self.model, epoch=epoch if batch_idx == 0 else None,
                    disable_pruner=disable_pruner,
                    disable_regrower=disable_regrower,
                    pruner_type=pruner_type,
                    use_anticryst=use_anticryst,
                    verbose=verbose_structural_logs,
                )
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

class LongTailedDataset(Dataset):
    """PyTorch Dataset for general Long-Tailed or Rare-Event image datasets.
    Supports loading images from individual files OR from an HDF5 archive."""
    def __init__(self, dataframe, image_dir, transform=None, hdf5_path=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self._hdf5_file = None
        self._error_printed = False

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
        if self.image_dir:
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    if not self._error_printed:
                        print(f"\n⚠️ Error loading image file {img_path}: {e}")
                        self._error_printed = True
                    image = None

        # Try 2: Load from HDF5 archive
        if image is None and self.hdf5_path and HAS_H5PY:
            try:
                hf = self._get_hdf5()
                if isic_id in hf:
                    img_bytes = hf[isic_id][()]
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                if not self._error_printed:
                    print(f"\n⚠️ Error loading image {isic_id} from HDF5: {e}")
                    self._error_printed = True
                image = None

        # Fallback: black placeholder
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        target = self.data_frame.iloc[idx]['target']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)


def get_imbalanced_dataloaders(batch_size=32, test_ratio=0.2, subsample_ratio=20, seed=42, allow_dummy_data=False):
    """
    Returns (train_loader, val_loader, cal_loader, test_loader, num_classes, cw, p_true, p_train).
    Uses stratified splitting to create train, val, cal, and test sets.
    Requires a real ISIC dataset by default. Dummy data is available only for
    explicit dry-runs with allow_dummy_data=True.
    """
    num_classes = 2

    csv_path = None
    image_dir = None
    
    # List of possible base directories to search for the dataset
    search_dirs = []
    if os.environ.get('ISIC_ROOT'):
        search_dirs.append(os.environ['ISIC_ROOT'])
    search_dirs.extend([
        r'E:\Testing\mdep\isic-2024-challenge',  # User local path
        './data/isic-2024-challenge',
        './data/isic2024',
        '/kaggle/input',                         # Kaggle root
        '/kaggle/input/competitions',            # Kaggle competitions
    ])
    
    def _try_find_dataset(base_dir):
        """Search for train-metadata.csv in a directory and return (csv_path, image_dir) or (None, None)."""
        if not os.path.isdir(base_dir):
            return None, None
            
        # Strategy 1: Check if the base_dir itself is the dataset root
        candidate_csv = os.path.join(base_dir, 'train-metadata.csv')
        if os.path.exists(candidate_csv):
            for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
                candidate_img = os.path.join(base_dir, img_sub)
                if os.path.isdir(candidate_img):
                    print(f"✅ Found ISIC dataset at: {base_dir}/")
                    return candidate_csv, candidate_img
            candidate_img = os.path.join(base_dir, 'train-image')
            print(f"✅ Found ISIC dataset at: {base_dir}/")
            return candidate_csv, candidate_img

        # Strategy 2: Check subdirectories (1 level deep)
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

    for search_dir in search_dirs:
        csv_path, image_dir = _try_find_dataset(search_dir)
        if csv_path is not None:
            break
            
    # Strategy 3: Recursive scan for Kaggle as a last resort
    if csv_path is None and os.path.isdir('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            depth = root.replace('/kaggle/input', '').count(os.sep)
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
        print(f"❌ train-metadata.csv not found in any search directories.")

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
        if not allow_dummy_data:
            raise FileNotFoundError(
                "ISIC dataset not found. Add the ISIC 2024 Kaggle competition "
                "input so train-metadata.csv and train-image.hdf5 are visible "
                "under /kaggle/input. Use allow_dummy_data=True only for local dry-runs."
            )
        print("⚠ ISIC dataset not found. Falling back to dummy data because allow_dummy_data=True.")
        X = torch.randn(200, 3, 224, 224)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, Y)
        tr = Subset(full, range(120))
        va = Subset(full, range(120, 140))
        ca = Subset(full, range(140, 160))
        te = Subset(full, range(160, 200))
        p_true = [0.5, 0.5]
        p_train = [0.5, 0.5]
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(va, batch_size=batch_size),
                DataLoader(ca, batch_size=batch_size),
                DataLoader(te, batch_size=batch_size),
                num_classes,
                torch.ones(num_classes),
                p_true,
                p_train)

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
    
    # Ensure patient_id exists and has no NaNs
    if 'patient_id' in df.columns:
        df = df.dropna(subset=['patient_id']).reset_index(drop=True)
        
        # Group by patient_id and get the max target for each patient to preserve class balance
        patient_df = df.groupby('patient_id')['target'].max().reset_index()
        
        # Split patients into train/test stratified by patient-level target
        train_patients, test_patients = train_test_split(
            patient_df, test_size=test_ratio, stratify=patient_df['target'], random_state=seed
        )
        
        # Split train patients into train/val (70/10 of total, which is 12.5% of train)
        train_patients, val_patients = train_test_split(
            train_patients, test_size=0.125, stratify=train_patients['target'], random_state=seed
        )
        
        # Split val patients in half to create validation and calibration hold-out sets (5% each)
        val_patients, cal_patients = train_test_split(
            val_patients, test_size=0.5, stratify=val_patients['target'], random_state=seed
        )
        
        # Map patients back to the original dataframe
        train_df = df[df['patient_id'].isin(train_patients['patient_id'])].reset_index(drop=True)
        val_df = df[df['patient_id'].isin(val_patients['patient_id'])].reset_index(drop=True)
        cal_df = df[df['patient_id'].isin(cal_patients['patient_id'])].reset_index(drop=True)
        test_df = df[df['patient_id'].isin(test_patients['patient_id'])].reset_index(drop=True)
    else:
        # Fallback if patient_id doesn't exist
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, stratify=df['target'], random_state=seed
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.125, stratify=train_df['target'], random_state=seed
        )
        val_df, cal_df = train_test_split(
            val_df, test_size=0.5, stratify=val_df['target'], random_state=seed
        )

    # Calculate true prior probabilities before subsampling
    class_counts_true = train_df['target'].value_counts().sort_index()
    total_true = len(train_df)
    p_true = [class_counts_true.get(c, 0) / total_true for c in range(num_classes)]

    # Subsample benign class to match malignant class * subsample_ratio
    if subsample_ratio is not None and subsample_ratio > 0:
        train_malignant = train_df[train_df['target'] == 1]
        train_benign = train_df[train_df['target'] == 0]
        num_train_malignant = len(train_malignant)
        if num_train_malignant > 0:
            num_benign_to_sample = min(len(train_benign), num_train_malignant * subsample_ratio)
            train_benign_sampled = train_benign.sample(n=num_benign_to_sample, random_state=seed)
            train_df = pd.concat([train_malignant, train_benign_sampled]).reset_index(drop=True)
            # Shuffle the combined dataframe
            train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            print(f"📉 Subsampled training set: {num_train_malignant} malignant, {len(train_benign_sampled)} benign.")

    print(f"📊 Train: {len(train_df)} samples  |  Val: {len(val_df)} samples  |  Cal: {len(cal_df)} samples  |  Test: {len(test_df)} samples")
    train_ds = LongTailedDataset(train_df, image_dir, transform=train_tf, hdf5_path=hdf5_path)
    val_ds   = LongTailedDataset(val_df,   image_dir, transform=test_tf,  hdf5_path=hdf5_path)
    cal_ds   = LongTailedDataset(cal_df,   image_dir, transform=test_tf,  hdf5_path=hdf5_path)
    test_ds  = LongTailedDataset(test_df,  image_dir, transform=test_tf,  hdf5_path=hdf5_path)
    
    import platform
    workers = 0 if platform.system() == 'Windows' else 4
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2 if workers > 0 else None)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, prefetch_factor=2 if workers > 0 else None)
    cal_loader   = DataLoader(cal_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, prefetch_factor=2 if workers > 0 else None)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, prefetch_factor=2 if workers > 0 else None)

    # Compute class weights (dampened inverse frequency to prevent loss/gradient explosion)
    import math
    class_counts = train_df['target'].value_counts().sort_index()
    total = len(train_df)
    cw_raw = [math.sqrt(total / class_counts.get(c, 1)) for c in range(num_classes)]
    majority_weight = cw_raw[0]
    cw = torch.tensor([w / majority_weight for w in cw_raw], dtype=torch.float32)
    print(f"⚖️  Class weights: {dict(enumerate(cw.tolist()))}")

    # Compute training class priors:
    p_train = [class_counts.get(c, 0) / total for c in range(num_classes)]

    return train_loader, val_loader, cal_loader, test_loader, num_classes, cw, p_true, p_train


def replace_conv2d_with_mdep(model):
    """Recursively swap nn.Conv2d / nn.Linear → MDEPConv2d / MDEPLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_conv2d_with_mdep(module)


# ============================================================================
#  SECTION 6 — Evaluation, Metrics & Visualization
# ============================================================================

def compute_adaptive_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error with adaptive (equal-frequency) bins."""
    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0, [0.0] * n_bins, [0.0] * n_bins, [0] * n_bins

    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    sorted_conf = confidences[sorted_indices]
    sorted_acc = accuracies[sorted_indices]
    
    bin_sizes = []
    bin_accs = []
    bin_confs = []
    ece = 0.0
    
    # Calculate step size
    step = n_samples / n_bins
    for i in range(n_bins):
        start_idx = int(i * step)
        end_idx = int((i + 1) * step) if i < n_bins - 1 else n_samples
        
        b_size = end_idx - start_idx
        if b_size == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            continue
            
        b_acc = sorted_acc[start_idx:end_idx].mean()
        b_conf = sorted_conf[start_idx:end_idx].mean()
        
        ece += (b_size / n_samples) * abs(b_acc - b_conf)
        bin_accs.append(b_acc)
        bin_confs.append(b_conf)
        bin_sizes.append(b_size)
        
    return ece, bin_accs, bin_confs, bin_sizes


def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error with equal-width bins (Standard ECE)."""
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


def compute_class_conditional_ece(probs, y_true, n_bins=15):
    """Computes Class-Conditional ECE for class 0 and class 1 using adaptive binning."""
    num_classes = probs.shape[1]
    class_eces = {}
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(float)
        probs_c = probs[:, c]
        ece_c, _, _, _ = compute_adaptive_ece(probs_c, y_true_c, n_bins)
        class_eces[c] = ece_c
    return class_eces


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
    
    n_samples = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    
    coverages = np.arange(1, n_samples + 1) / n_samples
    risks = cumulative_errors / np.arange(1, n_samples + 1)
        
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


def check_representational_collapse(model, detail=False):
    """
    Diagnoses Representational Collapse in 2:4 structured sparsity.
    A collapse occurs not when sparsity reaches 100% (which is impossible under 2:4),
    but when the Multi-Agent system loses its ability to rank connections, leading to random pruning.
    We check for:
    1. Score Variance Tracking (Zero Variance)
    2. Mean Drift Control (Negative Drift)
    3. Structural-gradient vitality for dynamic sparse runs.

    Static 2:4 baselines intentionally keep the score tensor fixed after mask
    initialization, so missing structural gradients are reported as N/A rather
    than representational collapse.
    """
    print("\n🔬 Representational Collapse Diagnostics")
    print("-" * 105)
    if detail:
        print(f"  {'Layer':30s} | {'Score Std':12s} | {'Score Mean':12s} | {'Struct Grad':12s} | {'Status'}")
        print("-" * 105)
    
    all_pass = True
    issue_counts = {}
    static_layers = 0
    total_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            total_layers += 1
            scores = module.scores.data
            std = scores.std().item() if scores.numel() > 1 else 0.0
            mean = scores.mean().item()
            
            grad_norm = 0.0
            grad_L = getattr(module, 'grad_L_w', None)
            if grad_L is not None:
                grad_norm = grad_L.norm().item()
            is_static_baseline = bool(getattr(module, 'static_24_baseline', False))
                
            status = "✅ PASS"
            issues = []
            if std <= 1e-4:
                issues.append("Zero Variance")
            if mean <= -10.0:
                issues.append("Negative Drift")
            if (not is_static_baseline) and grad_norm <= 1e-6:
                issues.append("Dead Gradient")
                
            if issues:
                status = "❌ FAIL (" + ", ".join(issues) + ")"
                all_pass = False
                for issue in issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
            elif is_static_baseline:
                status = "ℹ️ STATIC 2:4 (grad N/A)"
                static_layers += 1
                
            if detail:
                print(f"  {name:30s} | {std:12.4e} | {mean:12.4f} | {grad_norm:12.4e} | {status}")
            
    print("-" * 105)
    if not detail:
        issue_text = ", ".join(f"{key}: {value}" for key, value in sorted(issue_counts.items())) if issue_counts else "none"
        print(f"  Layers checked: {total_layers} | static grad-N/A layers: {static_layers} | issues: {issue_text}")
    if all_pass:
        print("  🌟 OVERALL STATUS: HEALTHY (No Representational Collapse Detected)")
    else:
        print("  ⚠️ OVERALL STATUS: WARNING (Representational Collapse Detected in some layers)")
    print()


def print_sparsity_report(model, detail=False):
    """Per-layer and total sparsity stats + 2:4 pattern check + MACs estimation."""
    print("\n📐 Sparsity & Hardware Metrics Report")
    print("-" * 75)
    total_params = 0
    total_zeros  = 0
    total_macs_dense = 0
    total_macs_sparse = 0
    valid_24 = 0
    checked_24 = 0
    
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
                checked_24 += 1
                valid_24 += int(valid)
                pattern = "✅ 2:4 (TensorCore Ready)" if valid else "❌ Not 2:4"
            else:
                pattern = "⚠ skip (size%4≠0)"
            if detail:
                print(f"  {name:30s} | {sparsity:5.1f}% sparse | {pattern}")
            
    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0
    print("-" * 75)
    print(f"  {'TOTAL PARAMS':30s} | {overall:5.1f}% sparse")
    print(f"  {'VALID 2:4 LAYERS':30s} | {valid_24}/{checked_24}")
    print(f"  {'THEORETICAL MACs SAVED':30s} | {macs_saved:5.1f}% reduction in MDEP layers")
    print("  *(Note: Ampere GPU Tensor Cores provide 2x speedup for strict 2:4 sparsity)*")
    print()
    
    # Run Advanced Collapse Diagnostics
    check_representational_collapse(model, detail=detail)


def compute_patient_level_se_top15(df, probs):
    """
    Computes patient-level sensitivity in top-15.
    """
    df = df.copy()
    df['prob'] = probs[:, 1]
    
    total_malignancies = 0
    found_malignancies = 0
    
    for patient_id, group in df.groupby('patient_id'):
        patient_malignancies = group['target'].sum()
        if patient_malignancies == 0:
            continue
            
        total_malignancies += patient_malignancies
        top_15 = group.sort_values(by='prob', ascending=False).head(15)
        found_malignancies += top_15['target'].sum()
        
    if total_malignancies == 0:
        return 1.0
    return found_malignancies / total_malignancies

def compute_isic_pauc(y_true, y_prob, min_tpr=0.80):
    """Official ISIC 2024 metric: pAUC above a given TPR threshold (e.g. 0.80)."""
    from sklearn.metrics import roc_curve, auc
    v_gt = abs(np.asarray(y_true) - 1)
    v_pred = -1.0 * np.asarray(y_prob)
    max_fpr = abs(1.0 - min_tpr)
    
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1.0:
        return auc(fpr, tpr)
        
    stop = np.searchsorted(fpr, max_fpr, 'right')
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr_at_max_fpr = np.interp(max_fpr, x_interp, y_interp)
    
    fpr = np.append(fpr[:stop], max_fpr)
    tpr = np.append(tpr[:stop], tpr_at_max_fpr)
    
    return auc(fpr, tpr)

def compute_aurc(y_true, y_pred, confidences):
    """Computes Area Under Risk-Coverage Curve (AURC)."""
    from sklearn.metrics import auc
    sorted_indices = np.argsort(-confidences)
    sorted_true = y_true[sorted_indices]
    sorted_pred = y_pred[sorted_indices]
    
    n_samples = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    
    coverages = np.arange(1, n_samples + 1) / n_samples
    risks = cumulative_errors / np.arange(1, n_samples + 1)
        
    return auc(coverages, risks)

@torch.no_grad()
def evaluate(model, val_loader, test_loader, device, num_classes, temperature=1.0, bias=None, plot=True):
    """Full evaluation: computes 10 metrics across multiple thresholds and returns (pauc, metrics)."""
    model.eval()
    from sklearn.metrics import fbeta_score

    # Use flat prior for evaluation since logits are corrected by PostHocPriorCorrection
    alpha_prior = None

    # Identify head and model components for temperature scaling
    is_resnet = hasattr(model, 'fc') and isinstance(model.fc, nn.Sequential)
    head = model.fc if is_resnet else model.head
    linear = head[0]
    evidence_layer = head[1]

    # Temporarily replace head with Identity to extract backbone features
    if is_resnet:
        model.fc = nn.Identity()
    else:
        model.head = nn.Identity()

    # 1. Collect validation predictions to optimize decision thresholds
    val_targets = []
    val_probs = []
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        features = model(inputs)
        logits = linear(features)
        
        # Apply temperature scaling and bias to logits
        logits_scaled = logits / temperature
        if bias is not None:
            logits_scaled = logits_scaled + bias
        evidence = evidence_layer(logits_scaled)
        
        unc = compute_uncertainties(evidence, alpha_prior=alpha_prior)
        p_hat = (unc['alpha'] / unc['S']).cpu().numpy()
        val_targets.append(targets.numpy())
        val_probs.append(p_hat)
        
    val_y_true = np.concatenate(val_targets)
    val_probs = np.concatenate(val_probs, axis=0)
    
    # Restore head
    if is_resnet:
        model.fc = head
    else:
        model.head = head
    
    # Threshold Optimization: Balanced Accuracy & Clinical (Sensitivity >= 80%, Max Specificity)
    best_t_bal_acc = 0.5
    best_bal_acc = 0.0
    
    best_t_clinical = 0.5
    best_spec_at_sens80 = 0.0
    found_sens80 = False
    
    thresholds = np.linspace(0.01, 0.99, 199)
    for t in thresholds:
        y_pred_t = (val_probs[:, 1] >= t).astype(int)
        
        # Balanced Accuracy
        bal_acc_t = balanced_accuracy_score(val_y_true, y_pred_t)
        if bal_acc_t > best_bal_acc:
            best_bal_acc = bal_acc_t
            best_t_bal_acc = t
            
        # Clinical Threshold
        tn, fp, fn, tp = confusion_matrix(val_y_true, y_pred_t, labels=[0, 1]).ravel()
        sens_t = tp / (tp + fn + 1e-8)
        spec_t = tn / (tn + fp + 1e-8)
        if sens_t >= 0.80:
            if spec_t > best_spec_at_sens80 or not found_sens80:
                best_spec_at_sens80 = spec_t
                best_t_clinical = t
                found_sens80 = True
                
    if not found_sens80:
        # Fallback to max sensitivity if >=80% is impossible
        best_sens = 0.0
        for t in thresholds:
            y_pred_t = (val_probs[:, 1] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_y_true, y_pred_t, labels=[0, 1]).ravel()
            sens_t = tp / (tp + fn + 1e-8)
            if sens_t > best_sens:
                best_sens = sens_t
                best_t_clinical = t

    print(f"[Validation] Optimized thresholds: Bal. Acc. = {best_t_bal_acc:.4f}, Clinical (Sens>=80%) = {best_t_clinical:.4f}")

    # Temporarily replace head with Identity to extract backbone features
    if is_resnet:
        model.fc = nn.Identity()
    else:
        model.head = nn.Identity()

    # 2. Collect test predictions
    all_targets  = []
    all_probs    = []
    all_u_e      = []
    all_u_a      = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        features = model(inputs)
        logits = linear(features)
        
        # Apply temperature scaling and bias to logits
        logits_scaled = logits / temperature
        if bias is not None:
            logits_scaled = logits_scaled + bias
        evidence = evidence_layer(logits_scaled)
        
        unc = compute_uncertainties(evidence, alpha_prior=alpha_prior)

        alpha = unc['alpha']
        S     = unc['S']
        p_hat = (alpha / S).cpu().numpy()

        all_targets.append(targets.numpy())
        all_probs.append(p_hat)
        all_u_e.append(unc['epistemic'].cpu().numpy()[:, 0])
        all_u_a.append(unc['aleatoric'].cpu().numpy()[:, 0])

    # Restore head
    if is_resnet:
        model.fc = head
    else:
        model.head = head

    y_true = np.concatenate(all_targets)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    
    confs = probs.max(axis=1)

    # 3. Compute threshold-independent metrics
    pauc = compute_isic_pauc(y_true, probs[:, 1], min_tpr=0.80)
    macro_auroc = roc_auc_score(y_true, probs[:, 1], average='macro')
    pr_auc = average_precision_score(y_true, probs[:, 1])
    
    # Compute ECE variants
    y_pred_default = probs.argmax(axis=1) if num_classes > 2 else (probs[:, 1] >= 0.5).astype(int)
    correct_default = (y_pred_default == y_true).astype(float)
    
    ece_adaptive, bin_accs, bin_confs, bin_sizes = compute_adaptive_ece(confs, correct_default)
    ece_eq_width, _, _, _ = compute_ece(confs, correct_default)
    class_eces = compute_class_conditional_ece(probs, y_true)
    
    # AURC is computed based on sorting confidences of predictions
    aurc = compute_aurc(y_true, y_pred_default, confs)

    # Helper function to compute threshold-dependent metrics
    def get_threshold_metrics(t):
        y_pred_t = (probs[:, 1] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal_acc = balanced_accuracy_score(y_true, y_pred_t)
        macro_f1 = f1_score(y_true, y_pred_t, average='macro')
        f2 = fbeta_score(y_true, y_pred_t, beta=2)
        return sens, spec, bal_acc, macro_f1, f2

    sens_05, spec_05, bal_acc_05, macro_f1_05, f2_05 = get_threshold_metrics(0.50)
    sens_bal, spec_bal, bal_acc_bal, macro_f1_bal, f2_bal = get_threshold_metrics(best_t_bal_acc)
    sens_clin, spec_clin, bal_acc_clin, macro_f1_clin, f2_clin = get_threshold_metrics(best_t_clinical)

    # Compute patient-level metrics
    if isinstance(test_loader.dataset, Subset) and hasattr(test_loader.dataset.dataset, 'data_frame'):
        test_df = test_loader.dataset.dataset.data_frame.iloc[test_loader.dataset.indices]
    elif hasattr(test_loader.dataset, 'data_frame'):
        test_df = test_loader.dataset.data_frame
    else:
        test_df = pd.DataFrame({
            'target': [test_loader.dataset[i][1].item() for i in range(len(test_loader.dataset))],
            'patient_id': [f"patient_{i // 5}" for i in range(len(test_loader.dataset))]
        })
    se_top15 = compute_patient_level_se_top15(test_df, probs)

    # Print results beautifully
    print("\n" + "="*80)
    print(" [EVAL] CLASSIFICATION METRICS PER DECISION THRESHOLD:")
    print("-"*80)
    print(f"{'Metric':25s} | {'Default (0.50)':16s} | {'Bal. Acc. Opt':16s} | {'Clinical Opt':16s}")
    print("-"*80)
    print(f"{'Decision Threshold':25s} | {0.5000:16.4f} | {best_t_bal_acc:16.4f} | {best_t_clinical:16.4f}")
    print(f"{'Sensitivity (Recall)':25s} | {sens_05:16.4f} | {sens_bal:16.4f} | {sens_clin:16.4f}")
    print(f"{'Specificity':25s} | {spec_05:16.4f} | {spec_bal:16.4f} | {spec_clin:16.4f}")
    print(f"{'Balanced Accuracy':25s} | {bal_acc_05:16.4f} | {bal_acc_bal:16.4f} | {bal_acc_clin:16.4f}")
    print(f"{'F2-Score':25s} | {f2_05:16.4f} | {f2_bal:16.4f} | {f2_clin:16.4f}")
    print(f"{'Macro F1-Score':25s} | {macro_f1_05:16.4f} | {macro_f1_bal:16.4f} | {macro_f1_clin:16.4f}")
    print("-"*80)
    print("\n [EVAL] CALIBRATION & RANKING METRICS (THRESHOLD-INDEPENDENT):")
    print("-"*80)
    print(f"  pAUC 0.80 (ISIC 2024)      : {pauc:.4f}")
    print(f"  SE_top-15 (Patient-level)  : {se_top15:.4f}")
    print(f"  PR-AUC                     : {pr_auc:.4f}")
    print(f"  Macro-AUROC                : {macro_auroc:.4f}")
    print(f"  AURC (Risk-Coverage)       : {aurc:.4f}")
    print(f"  ECE (Adaptive, 15 bins)    : {ece_adaptive:.4f}")
    print(f"  ECE (Equal-Width, 15 bins) : {ece_eq_width:.4f}")
    print(f"  Class-Cond. ECE - Class 0  : {class_eces[0]:.4f}")
    print(f"  Class-Cond. ECE - Class 1  : {class_eces[1]:.4f}")
    print(f"  Mean Epistemic uncertainty : {u_e.mean():.4f}")
    print(f"  Mean Aleatoric uncertainty : {u_a.mean():.4f}")
    print("="*80 + "\n")

    if plot:
        plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
        plot_uncertainty_histogram(u_e[correct_default.astype(bool)], u_e[~correct_default.astype(bool)])
        plot_pr_curve(y_true, probs[:, 1])
        plot_risk_coverage_curve(y_true, (probs[:, 1] >= best_t_clinical).astype(int), confs)

    metrics = {
        'pauc': pauc,
        'se_top15': se_top15,
        'pr_auc': pr_auc,
        'macro_auroc': macro_auroc,
        'aurc': aurc,
        'ece_adaptive': ece_adaptive,
        'ece_eq_width': ece_eq_width,
        'class_ece_0': class_eces[0],
        'class_ece_1': class_eces[1],
        'mean_epistemic': u_e.mean(),
        'mean_aleatoric': u_a.mean(),
        
        # Default threshold metrics
        'sens_default': sens_05,
        'spec_default': spec_05,
        'bal_acc_default': bal_acc_05,
        'f2_default': f2_05,
        'macro_f1_default': macro_f1_05,
        
        # Balanced Accuracy threshold metrics
        'sens_opt_bal': sens_bal,
        'spec_opt_bal': spec_bal,
        'bal_acc_opt_bal': bal_acc_bal,
        'f2_opt_bal': f2_bal,
        'macro_f1_opt_bal': macro_f1_bal,
        'best_threshold_bal': best_t_bal_acc,
        
        # Clinical threshold metrics
        'sens_opt_clinical': sens_clin,
        'spec_opt_clinical': spec_clin,
        'bal_acc_opt_clinical': bal_acc_clin,
        'f2_opt_clinical': f2_clin,
        'macro_f1_opt_clinical': macro_f1_clin,
        'best_threshold_clinical': best_t_clinical
    }

    return pauc, metrics




def calibrate_temperature(model, val_loader, device, p_true=None, p_train=None):
    """
    Calibrates the temperature parameter T on the validation set using L-BFGS,
    and dynamically optimizes decision thresholds for clinical support.
    """
    model.eval()
    logits_list = []
    labels_list = []
    
    # Identify model head
    is_resnet = hasattr(model, 'fc') and isinstance(model.fc, nn.Sequential)
    head = model.fc if is_resnet else model.head
    linear = head[0]
    evidence_layer = head[1]
    
    # Temporarily replace head with Identity to get features
    if is_resnet:
        model.fc = nn.Identity()
    else:
        model.head = nn.Identity()
        
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            features = model(inputs)
            logits = linear(features)
            logits_list.append(logits)
            labels_list.append(targets)
            
    # Restore original head
    if is_resnet:
        model.fc = head
    else:
        model.head = head
        
    logits = torch.cat(logits_list, dim=0).detach()
    labels = torch.cat(labels_list, dim=0).to(device)
    if p_true is not None and p_train is not None:
        prior_delta = torch.tensor(
            [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(logits.shape[1])],
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        prior_delta = torch.zeros(logits.shape[1], dtype=logits.dtype, device=logits.device)
    logits_for_calibration = logits + prior_delta

    def evidential_nll(scaled_logits):
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        probs = (unc['alpha'] / unc['S']).clamp_min(1e-8)
        return F.nll_loss(torch.log(probs), labels)
    
    # Optimize temperature parameter T and bias parameter b (Bias-Corrected TS)
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    bias = nn.Parameter(torch.zeros(logits.shape[1], device=device))
    optimizer = optim.LBFGS([temperature, bias], lr=0.01, max_iter=50)
    
    def eval_loss():
        optimizer.zero_grad()
        model.zero_grad(set_to_none=True)
        scaled_logits = logits_for_calibration / temperature.clamp_min(0.1) + bias
        loss = evidential_nll(scaled_logits)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    
    T = max(0.1, temperature.item())
    b = bias.detach()
    print(f"[INFO] Calibrated Temperature T: {T:.4f}, Prior Delta: {prior_delta.cpu().numpy()}, Bias: {b.cpu().numpy()}")
    
    # Optimize thresholds dynamically on calibrated validation predictions
    scaled_logits = logits_for_calibration / T + b
    with torch.no_grad():
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        val_probs = (unc['alpha'] / unc['S']).cpu().numpy()
        
    val_y_true = labels.cpu().numpy()
    
    best_t_bal_acc = 0.5
    best_bal_acc = 0.0
    best_t_clinical = 0.5
    best_spec_at_sens80 = 0.0
    found_sens80 = False
    
    thresholds = np.linspace(0.01, 0.99, 199)
    for t in thresholds:
        y_pred_t = (val_probs[:, 1] >= t).astype(int)
        bal_acc_t = balanced_accuracy_score(val_y_true, y_pred_t)
        if bal_acc_t > best_bal_acc:
            best_bal_acc = bal_acc_t
            best_t_bal_acc = t
            
        tn, fp, fn, tp = confusion_matrix(val_y_true, y_pred_t).ravel()
        sens_t = tp / (tp + fn + 1e-8)
        spec_t = tn / (tn + fp + 1e-8)
        if sens_t >= 0.80:
            if spec_t > best_spec_at_sens80 or not found_sens80:
                best_spec_at_sens80 = spec_t
                best_t_clinical = t
                found_sens80 = True
                
    if not found_sens80:
        best_sens = 0.0
        for t in thresholds:
            y_pred_t = (val_probs[:, 1] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_y_true, y_pred_t).ravel()
            sens_t = tp / (tp + fn + 1e-8)
            if sens_t > best_sens:
                best_sens = sens_t
                best_t_clinical = t
                
    print(f"[INFO] Dynamically optimized calibrated thresholds: Balanced = {best_t_bal_acc:.4f}, Rule-out (Sens>=80%) = {best_t_clinical:.4f}")
    
    thresholds_dict = {
        'rule_out': best_t_clinical,
        'high_recall': best_t_clinical,
        'double_read': best_t_clinical,
        'balanced': best_t_bal_acc,
        'rule_in': 0.5000  # Rule-in is fixed at 0.5000 to maximize specificity
    }
    
    return T, b, thresholds_dict


class AdaptiveThresholdDecisionSupport(nn.Module):
    def __init__(self, model, temperature=1.0, bias=None, is_resnet=False, true_class_prior=[0.9985, 0.0015], train_class_prior=[0.95238, 0.04762], thresholds=None):
        super().__init__()
        self.model = model
        self.is_resnet = is_resnet
        self.temperature = temperature
        self.bias = bias
        self.p_train = train_class_prior
        self.p_true = true_class_prior
        
        # Save original head and replace with Identity
        self.original_head = self.model.fc if is_resnet else self.model.head
        if is_resnet:
            self.model.fc = nn.Identity()
        else:
            self.model.head = nn.Identity()
            
        self.linear = self.original_head[0]
        self.evidence_layer = self.original_head[1]
        
        # Save original logit adjustment of evidence layer and temporarily clear it
        if hasattr(self.evidence_layer, 'logit_adjustment'):
            self.original_logit_adjustment = self.evidence_layer.logit_adjustment.clone()
            self.evidence_layer.logit_adjustment = torch.zeros(1, device=self.evidence_layer.logit_adjustment.device)
        else:
            self.original_logit_adjustment = None

        # Dynamically optimized thresholds or default fallback
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = {
                'high_recall': 0.1882,
                'balanced': 0.2129,
                'rule_in': 0.5000
            }
        
    def restore_model(self):
        """Restore model's original head."""
        if self.is_resnet:
            self.model.fc = self.original_head
        else:
            self.model.head = self.original_head
        # Restore logit adjustment
        if self.original_logit_adjustment is not None and hasattr(self.evidence_layer, 'logit_adjustment'):
            self.evidence_layer.logit_adjustment = self.original_logit_adjustment
            
    def forward(self, x, mode="balanced", quality_gated=False):
        # 1. Trích xuất đặc trưng từ backbone
        features = self.model(x)
        
        # 2. Tính logits tuyến tính
        logits = self.linear(features)
        
        # 3. Apply PostHocPriorCorrection on logits
        # (This implements the prior adjustment directly)
        p_true_tensor = torch.tensor(self.p_true, dtype=torch.float32, device=logits.device)
        p_train_tensor = torch.tensor(self.p_train, dtype=torch.float32, device=logits.device)
        delta = torch.log(p_true_tensor + 1e-8) - torch.log(p_train_tensor + 1e-8)
        logits = logits + delta

        alpha_prior_active = torch.ones(2, dtype=torch.float32, device=logits.device)
            
        # 4. Hiệu chuẩn bằng Bias-Corrected Temperature Scaling
        logits_scaled = logits / self.temperature
        if self.bias is not None:
            logits_scaled = logits_scaled + self.bias
            
        evidence = self.evidence_layer(logits_scaled)
        
        # 5. Tính toán các độ bất định và xác suất
        unc = compute_uncertainties(evidence, alpha_prior=alpha_prior_active)
        probs = unc['alpha'] / unc['S']
        assert torch.all((probs >= 0.0) & (probs <= 1.0)), "Probability bounds violated [0, 1]!"
        u_e = unc['epistemic']
        u_a = unc['aleatoric']
        
        # 7. Adaptive Thresholding and Selective Prediction
        t = self.thresholds.get(mode, 0.5000)
        pred = (probs[:, 1] >= t).long()
        
        # Apply Epistemic Uncertainty Filter (High-Recall / Fail-Safe)
        # 0: Majority Class, 1: Rare Class, 2: Flagged for Human Review, 3: Low Quality/OOD
        final_decision = pred.clone()
        
        if mode in ["double_read", "high_recall"]:
            is_majority_class = (pred == 0)
            is_uncertain = (u_e.squeeze(-1) >= 0.30)
            final_decision[is_majority_class & is_uncertain] = 2
            
        # Apply Aleatoric Uncertainty Filter (Quality-Gated)
        if quality_gated:
            is_uncertain_sample = (u_a.squeeze(-1) >= 0.45)
            final_decision[is_uncertain_sample] = 3
            
        return final_decision, probs, u_e, u_a


@torch.no_grad()
def evaluate_adaptive_modes(decision_support, test_loader, device):
    """
    Evaluates the 3 adaptive operating modes on the Test set and prints a comparison table.
    """
    print("\n" + "="*80)
    print(" [ADAPTIVE MODES] EVALUATION OF ADAPTIVE OPERATING POINTS ON TEST SET:")
    print("-"*80)
    print(f"{'Adaptive Mode':20s} | {'Sensitivity':12s} | {'Specificity':12s} | {'Review %':12s} | {'Discard %':12s}")
    print("-"*80)
    
    # 1. Balanced Mode
    targets_all, preds_all = [], []
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        final_dec, _, _, _ = decision_support(inputs, mode="balanced")
        targets_all.append(targets.numpy())
        preds_all.append(final_dec.cpu().numpy())
    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(preds_all)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    print(f"{'Balanced Utility':20s} | {sens:12.4f} | {spec:12.4f} | {'0.00%':12s} | {'0.00%':12s}")
    
    # 2. High-Recall / Fail-Safe (Sens >= 95%)
    targets_all, preds_all = [], []
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        final_dec, _, _, _ = decision_support(inputs, mode="high_recall")
        targets_all.append(targets.numpy())
        preds_all.append(final_dec.cpu().numpy())
    y_pred = np.concatenate(preds_all)
    
    # In fail-safe mode, class 2 represents referral for human review.
    referred = (y_pred == 2).sum()
    referred_pct = referred / len(y_pred) * 100
    
    valid_mask = (y_pred != 2)
    if valid_mask.sum() > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    else:
        sens, spec = 1.0, 0.0
    print(f"{'High-Recall (Fail-Safe)':20s} | {sens:12.4f} | {spec:12.4f} | {referred_pct:11.2f}% | {'0.00%':12s}")
    
    # 3. Quality-Gated Mode
    targets_all, preds_all = [], []
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        final_dec, _, _, _ = decision_support(inputs, mode="balanced", quality_gated=True)
        targets_all.append(targets.numpy())
        preds_all.append(final_dec.cpu().numpy())
    y_pred = np.concatenate(preds_all)
    
    discarded = (y_pred == 3).sum()
    discarded_pct = discarded / len(y_pred) * 100
    
    valid_mask = (y_pred != 3)
    if valid_mask.sum() > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    else:
        sens, spec = 0.0, 0.0
    print(f"{'Quality-Gated Mode':20s} | {sens:12.4f} | {spec:12.4f} | {'0.00%':12s} | {discarded_pct:11.2f}%")
    print("="*80 + "\n")


# ============================================================================
#  SECTION 7 — main()
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GUDS-EDL Core Training")
    parser.add_argument('--disable_pruner', action='store_true', help="Disable Microglia pruning")
    parser.add_argument('--disable_regrower', action='store_true', help="Disable Astrocyte regrowing")
    parser.add_argument('--pruner_type', type=str, default='signed_first_order', choices=['signed_first_order', 'absolute_grad', 'magnitude', 'random'])
    parser.add_argument('--regrower_type', type=str, default='kl_uniform', choices=['kl_uniform', 'class_conditioned', 'gradient', 'random'])
    parser.add_argument('--kl_scaling', type=str, default='asymmetric', choices=['asymmetric', 'symmetric'])
    parser.add_argument('--disable_efl', action='store_true', help="Disable Evidential Focal Loss weighting")
    parser.add_argument('--disable_anticryst', action='store_true', help="Disable Astrocyte anti-crystallization")
    args = parser.parse_args()
    args.use_anticryst = not args.disable_anticryst
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")
    print(f"⚙️  Ablation Configs: {vars(args)}")

    # ── Data (stratified train / test split) ────────────────────────
    train_loader, val_loader, cal_loader, test_loader, num_classes, class_weights, p_true, p_train = get_imbalanced_dataloaders(batch_size=32)
    print(f"📊 Classes: {num_classes}")
    print(f"   Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}  |  Cal batches: {len(cal_loader)}  |  Test batches: {len(test_loader)}")

    # ── Model: ResNet-18 with EDL head ──────────────────────────────
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        EvidenceLayer(activation='softplus')
    )
    # Initialize evidence output to be small to prevent KL explosion
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    replace_conv2d_with_mdep(model)
    model = model.to(device)
    has_isic = os.path.exists('/kaggle/input') or os.path.exists(r'E:\Testing\mdep\isic-2024-challenge')
    total_epochs  = 40 if has_isic else 3
    warmup_epochs = 12 if has_isic else 1

    # ── Optimizer & Loss ───────────────────────────────────────────
    criterion = EvidentialFocalLoss(
        gamma=1.2, num_classes=num_classes, kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=warmup_epochs, total_epochs=total_epochs,
        disable_efl=args.disable_efl, kl_scaling=args.kl_scaling
    )
    # Khắc phục lỗi Optimizer Hijacking: chặn 'scores' khỏi AdamW
    trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
    optimizer = optim.Adam(trainable_params, lr=4.0e-05)
    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs, args=args)

    # Ensure output directory exists (Kaggle writable path)
    output_dir = '/kaggle/working/' if os.path.exists('/kaggle/working/') else './'
    
    checkpoint_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    best_checkpoint_path = os.path.join(output_dir, 'best_checkpoint.pth')
    start_epoch = 0
    best_loss = float('inf')
    start_time = time.time()

    # ── Initialize WandB ───────────────────────────────────────────
    has_wandb = False
    try:
        import wandb
        print("🔑 Logging into WandB...")
        if os.environ.get("WANDB_API_KEY"):
            wandb.login()
        else:
            wandb.login(anonymous="allow")
        
        wandb.init(
            project="MDEP-Microglial-Driven-Evidential-Pruning",
            name="MDEP-Main-Run",
            config={
                "learning_rate": 4.0e-05,
                "total_epochs": total_epochs,
                "warmup_epochs": warmup_epochs,
                "batch_size": 32,
                "architecture": "ResNet-18-MDEP"
            }
        )
        has_wandb = True
    except Exception as e:
        print(f"⚠️ WandB login/init skipped: {e}. Running without online logging.")

    # Look for checkpoint to resume from
    resume_path = checkpoint_path
    if not os.path.exists(resume_path) and os.path.exists('/kaggle/input'):
        # Fallback to searching Kaggle input directories if no checkpoint is in working dir
        found_checkpoint = None
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'latest_checkpoint.pth' in files:
                found_checkpoint = os.path.join(root, 'latest_checkpoint.pth')
                break  # Prefer latest over best
            elif 'best_checkpoint.pth' in files and not found_checkpoint:
                found_checkpoint = os.path.join(root, 'best_checkpoint.pth')
                
        if found_checkpoint:
            resume_path = found_checkpoint
            print(f"🔍 Found previous checkpoint in dataset: {resume_path}")

    if os.path.exists(resume_path):
        print(f"🔄 Found checkpoint at {resume_path}. Resuming training...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and hasattr(trainer, 'scaler'):
                trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"⏩ Resuming from epoch {start_epoch + 1} (Best Loss: {best_loss:.4f})")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting from scratch.")

    # ── Training ───────────────────────────────────────────────────
    print("\n🚀 Starting Training (MDEP Framework)")
    print("=" * 60)
    for epoch in range(start_epoch, total_epochs):
        # Time-out check (Kaggle T4 limit is 9 hours, we stop early at 8.2 hours = 29500s to save outputs gracefully)
        if time.time() - start_time > 29500:
            print("⏳ Approaching Kaggle 9-hour limit. Stopping training early to save checkpoints gracefully!")
            break

        loss = trainer.train_epoch(epoch, train_loader, device)
        phase = "Warm-up (Dense)" if epoch < warmup_epochs else "Dynamic 2:4 Sparsity"
        gamma = trainer.step_gamma(epoch)
        print(
            f"  Epoch [{epoch+1:>2}/{total_epochs}]  "
            f"| Phase: {phase:<22} "
            f"| γ: {gamma:.4f}  "
            f"| Loss: {loss:.4f}"
        )
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': trainer.scaler.state_dict() if hasattr(trainer, 'scaler') and trainer.scaler is not None else None,
            'loss': loss,
            'best_loss': best_loss
        }
        torch.save(checkpoint, checkpoint_path)
        if has_wandb:
            try:
                wandb.save(checkpoint_path)
            except Exception as e:
                print(f"⚠️ Failed to upload checkpoint to WandB: {e}")
        
        # Save best checkpoint
        if loss < best_loss:
            best_loss = loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, best_checkpoint_path)
            print(f"⭐ New best loss: {best_loss:.4f}. Saved best checkpoint.")
            if has_wandb:
                try:
                    wandb.save(best_checkpoint_path)
                except Exception as e:
                    print(f"⚠️ Failed to upload best checkpoint to WandB: {e}")
                    
        if has_wandb:
            try:
                wandb.log({
                    "epoch": epoch + 1,
                    "loss": loss,
                    "gamma": gamma,
                    "best_loss": best_loss,
                    "phase_idx": 0 if epoch < warmup_epochs else 1
                })
            except Exception as e:
                print(f"⚠️ Failed to log metrics to WandB: {e}")
            
    print("=" * 60)
    print("✅ Training complete.\n")

    # Load best checkpoint weights if available before final evaluation & save
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from {best_checkpoint_path} for final evaluation...")
        try:
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
        except Exception as e:
            print(f"⚠️ Error loading best checkpoint: {e}. Evaluating with final weights.")

    # 1. Calibrate temperature and optimize thresholds dynamically on Calibration Hold-out set
    temperature, bias, thresholds = calibrate_temperature(model, cal_loader, device, p_true=p_true, p_train=p_train)

    # 2. Wrap model in Adaptive Decision Support using calibrated temperature and thresholds
    decision_support = AdaptiveThresholdDecisionSupport(
        model, is_resnet=True, thresholds=thresholds, temperature=temperature, bias=bias,
        true_class_prior=p_true, train_class_prior=p_train
    )

    # 3. Evaluate adaptive operating modes on the Test set and display comparison table
    evaluate_adaptive_modes(decision_support, test_loader, device)

    # 4. Save final calibrated model checkpoint with temperature and thresholds metadata
    calibrated_checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': temperature,
        'bias': bias,
        'thresholds': thresholds,
        'p_true': p_true,
        'p_train': p_train
    }
    calib_checkpoint_path = os.path.join(output_dir, 'resnet_calibrated_adaptive.pth')
    torch.save(calibrated_checkpoint, calib_checkpoint_path)
    print(f"[INFO] Saved final calibrated checkpoint to '{calib_checkpoint_path}'")

    # 5. Restore model structure back to its original state for traditional evaluations
    decision_support.restore_model()

    # ── Evaluation ─────────────────────────────────────────────────
    prior_delta = torch.tensor(
        [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    if hasattr(model.fc[1], 'logit_adjustment'):
        model.fc[1].logit_adjustment = torch.zeros(1, dtype=torch.float32, device=device)
        print(f"⚖️ Applied explicit evaluation prior delta: {prior_delta.cpu().numpy()}")
    
    _, eval_metrics = evaluate(model, val_loader, test_loader, device, num_classes, temperature=temperature, bias=eval_bias, plot=True)
    print_sparsity_report(model)
    
    if has_wandb:
        try:
            wandb.log(eval_metrics)
            wandb.finish()
        except Exception as e:
            print(f"⚠️ Failed to upload final evaluation to WandB: {e}")

    # ── Saving Model ───────────────────────────────────────────────
    model_save_path = os.path.join(output_dir, 'mdep_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print("=" * 60)
    print(f"💾 Tải trọng số mô hình đã được lưu tại: {model_save_path}")
    print("   (Bạn có thể tải file này về từ tab 'Output' trên Kaggle)")

# ============================================================================
#  SECTION 8 — Ablation Study Harness (uncomment to run)
# ============================================================================
#
def update_scores_ablation(model, beta=1.0, mode='full'):
    """
    Ablation wrapper around update_scores_agents.
      mode='full'       → both Microglia + Astrocyte (default)
      mode='prune_only' → only Microglia scoring (G_ij = 0)
      mode='grow_only'  → only Astrocyte scoring (C_ij = 0)
    """
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                continue
            w_val = module.weight.data
            # Microglia
            c1 = torch.abs(w_val * module.grad_L_w)
            c1_min = c1.min()
            c1_max = c1.max()
            c1_norm = (c1 - c1_min) / (c1_max - c1_min + 1e-8)
            c2 = torch.abs(w_val * getattr(module, 'grad_ua_w', torch.zeros_like(w_val)))
            c2_min = c2.min()
            c2_max = c2.max()
            c2_norm = (c2 - c2_min) / (c2_max - c2_min + 1e-8)
            C_ij = c1_norm + beta * c2_norm
            # Astrocyte
            u_e_node = getattr(module, 'u_e_node', None)
            if u_e_node is not None:
                if isinstance(module, MDEPLinear):
                    g1 = u_e_node.unsqueeze(1).expand_as(w_val)
                elif isinstance(module, MDEPConv2d):
                    g1 = u_e_node.view(-1, 1, 1, 1).expand_as(w_val)
                else:
                    g1 = torch.zeros_like(w_val)
            else:
                g1 = torch.zeros_like(w_val)
            g1_norm = g1 / (g1.max() + 1e-8)
            g2 = torch.abs(module.grad_L_w)
            g2_norm = g2 / (g2.max() + 1e-8)
            G_ij = g1_norm * g2_norm
            # Apply ablation
            if mode == 'prune_only':
                G_ij = torch.zeros_like(G_ij)
            elif mode == 'grow_only':
                C_ij = torch.zeros_like(C_ij)
            delta_S = C_ij + G_ij
            beta_m = 0.9
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            eta = 0.1
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            module.scores.data.sub_(module.scores.data.mean())
#
# # To run an ablation study, uncomment and call:
# # for mode in ['prune_only', 'grow_only', 'full']:
# #     print(f"\n{'='*60}\n  ABLATION: {mode}\n{'='*60}")
# #     <rebuild model, train with update_scores_ablation(..., mode=mode), evaluate>


# ── Run ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
