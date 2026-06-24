r"""
============================================================================
  MDEP — Microglial-Driven Evidential Pruning (Swin-T Backbone Version)
  Single-file Kaggle Notebook version
  
  HOW TO RUN ON KAGGLE:
    1. Create a new Notebook, set Accelerator to GPU (T4 or P100).
    2. Click "Add Data" → search "ISIC 2024" → add the challenge dataset.
    3. Copy-paste this entire file into a single code cell.
    4. Run the cell.
============================================================================
r"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
import wandb
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, precision_recall_curve, auc
)
from PIL import Image
import io

# ============================================================================
#  SECTION 1 — EDL Core (Evidential Deep Learning foundations)
# ============================================================================

class LogPriorCorrection(nn.Module):
    def __init__(self, p_true, p_train):
        super(LogPriorCorrection, self).__init__()
        p_true_tensor = torch.tensor(p_true, dtype=torch.float32)
        p_train_tensor = torch.tensor(p_train, dtype=torch.float32)
        delta = torch.log(p_true_tensor + 1e-8) - torch.log(p_train_tensor + 1e-8)
        self.register_buffer('delta', delta)

    def forward(self, logits):
        return logits + self.delta


class EvidenceLayer(nn.Module):
    r"""
    Ensures the output of the network is non-negative evidence (e >= 0).
    Replaces the traditional Softmax layer for EDL.
    r"""
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
    r"""
    Computes epistemic and aleatoric uncertainties from the Dirichlet evidence.
    
    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)
        
    Returns:
        dict: Epistemic uncertainty, aleatoric uncertainty, alpha, and Dirichlet strength S.
    r"""
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = evidence.shape[1]

    # Epistemic Uncertainty (due to lack of knowledge): u_e = K / S
    u_e = K / S

    # Aleatoric Uncertainty (due to inherent data noise):
    # u_a = sum_c (alpha_c / S) * (psi(S+1) - psi(alpha_c+1))
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
#  SECTION 2 — Loss Functions (Evidential Focal Loss + KL Distillation)
# ============================================================================

def kl_divergence(alpha, num_classes):
    r"""KL divergence between Dirichlet(alpha) and uniform Dirichlet(1,...,1).r"""
    beta = torch.ones(1, num_classes, dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def dirichlet_kl_divergence(alpha_s, alpha_t):
    r"""
    Computes the Kullback-Leibler divergence between two Dirichlet distributions
    represented by concentration parameters alpha_s (student) and alpha_t (teacher).
    r"""
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


class EvidentialFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=2, kl_lambda=0.1, class_weights=None, annealing_epochs=10, warmup_epochs=5, total_epochs=30):
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
        
        # Cross entropy term under Evidential Deep Learning
        loss_ce = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        # Dynamic gamma scheduling
        if epoch is not None:
            if epoch < self.warmup_epochs:
                gamma_val = 0.0
            elif epoch < self.warmup_epochs + 5:
                gamma_val = self.base_gamma * (epoch - self.warmup_epochs) / 5.0
            else:
                gamma_val = self.base_gamma
        else:
            gamma_val = self.gamma
            
        self.gamma = gamma_val
        
        # Focal weight modulation
        p_target = torch.sum(targets * p_hat, dim=1, keepdim=True)
        focal_weight = (1.0 - p_target.detach()) ** gamma_val
        
        if self.class_weights is not None:
            sample_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
        else:
            sample_weight = 1.0
            
        # KL regularizer to shrink evidence for incorrect classes
        alpha_tilde = targets + (1 - targets) * alpha
        loss_kl = kl_divergence(alpha_tilde, self.num_classes)
        
        if epoch is not None and self.annealing_epochs > 0:
            annealing_coef = min(1.0, epoch / self.annealing_epochs)
        else:
            annealing_coef = 1.0
            
        loss = sample_weight * (focal_weight * loss_ce + self.kl_lambda * annealing_coef * loss_kl)
        return torch.mean(loss)


# ============================================================================
#  SECTION 3 — MDEP Multi-Agent Sparsity Engine
# ============================================================================

class SmoothedSTE(torch.autograd.Function):
    r"""
    Smoothed Straight-Through Estimator with Local 2:4 Bounds.
    Forward: passes the hard binary mask unchanged, but utilizes precomputed local thresholds.
    Backward: approximates dM/dS ≈ sigma'((S - tau)/gamma) so gradients flow
              only to connections near the 2:4 survival boundary.
    r"""
    @staticmethod
    def forward(ctx, scores, mask, tau, gamma):
        ctx.save_for_backward(scores, tau, torch.tensor(gamma))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, tau, gamma = ctx.saved_tensors
        gamma_val = gamma.item()
        
        # Localized STE: margin to the boundary
        margin = scores - tau
        
        sig = torch.sigmoid(margin / gamma_val)
        grad_scores = grad_output * (sig * (1.0 - sig) / gamma_val + 0.05)
        return grad_scores, None, None, None


def generate_2_4_mask(scores):
    r"""Generates an NVIDIA 2:4 structured sparsity mask.r"""
    if scores.numel() % 4 != 0:
        return torch.ones_like(scores)
        
    shape = scores.shape
    scores_flat = scores.view(-1, 4)
    _, indices = torch.topk(scores_flat, 2, dim=-1)
    mask_flat = torch.zeros_like(scores_flat)
    mask_flat.scatter_(1, indices, 1.0)
    return mask_flat.view(shape)


class MDEPLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MDEPLinear, self).__init__(in_features, out_features, bias)
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight), persistent=False)
        self.register_buffer('tau', torch.zeros_like(self.weight), persistent=False)
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.tau, self.gamma)
            effective_weight = self.weight * differentiable_mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        self.__dict__['effective_weight'] = effective_weight
            
        return F.linear(x, effective_weight, self.bias)


class MDEPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MDEPConv2d, self).__init__(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight), persistent=False)
        self.register_buffer('tau', torch.zeros_like(self.weight), persistent=False)
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.tau, self.gamma)
            effective_weight = self.weight * differentiable_mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        self.__dict__['effective_weight'] = effective_weight
            
        return F.conv2d(x, effective_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def update_scores_agents(model, beta=1.0):
    r"""
    Updates latent scores S_ij using Microglia (pruning) and Astrocyte (growing) signals.
    Employs the corrected opposing forces: delta_S = G_ij - C_ij.
    r"""
    total_flips = 0
    total_elements = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                continue
            
            old_mask = module.mask.clone()
            w_val = module.weight.data
            
            # --- 1. Microglia agent: Pruning Signal (C_ij) ---
            c1 = torch.abs(w_val * module.grad_L_w)
            grad_ua_w = getattr(module, 'grad_ua_w', torch.zeros_like(w_val))
            c2 = torch.abs(w_val * grad_ua_w)
            
            c1_norm = torch.tanh(c1 / (c1.median() + 1e-8))
            c2_norm = torch.tanh(c2 / (c2.median() + 1e-8))
            C_ij = c1_norm + beta * c2_norm
            
            # --- 2. Astrocyte agent: Growing Signal (G_ij) ---
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
            
            g1_norm = torch.tanh(g1 / (g1.mean() + 1e-8))
            
            g2 = torch.abs(module.grad_L_w)
            g2_norm = torch.tanh(g2 / (g2.mean() + 1e-8))
            
            G_ij = g1_norm * g2_norm
            
            # --- Anti-Crystallization (Stochastic exploration) ---
            if G_ij.max().item() <= 1e-8:
                noise = 0.0316 * torch.randn_like(G_ij) * g1_norm
                G_ij = G_ij + torch.clamp(noise, min=0.0)
                
            # --- 3. Dynamic Opposing Forces (Delta S_ij) ---
            delta_S = G_ij - C_ij
            
            beta_m = 0.95
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            
            eta = 0.02
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            module.scores.data.sub_(module.scores.data.mean())
            module.scores.data.clamp_(min=-5.0, max=5.0)
            
            new_mask = generate_2_4_mask(module.scores.data)
            module.mask.copy_(new_mask)
            
            if module.scores.numel() % 4 == 0:
                scores_flat = module.scores.data.view(-1, 4)
                sorted_scores, _ = torch.sort(scores_flat, dim=-1, descending=True)
                s2 = sorted_scores[:, 1]
                s3 = sorted_scores[:, 2]
                tau = ((s2 + s3) / 2.0).unsqueeze(-1).expand_as(scores_flat).reshape(module.scores.shape)
                module.tau.copy_(tau)
            else:
                module.tau.zero_()
            
            flips = (old_mask != new_mask).sum().item()
            total_flips += flips
            total_elements += old_mask.numel()
            
    flop_rate = total_flips / (total_elements + 1e-8)
    return flop_rate


# ============================================================================
#  SECTION 4 — Swin Layer Replacement Utilities
# ============================================================================

def replace_swin_linear_with_mdep(model):
    r"""Recursively replaces nn.Linear inside Swin features to MDEPLinear.r"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None)
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Conv2d):
            # Skip the first stem Conv2d to protect patch representation
            if module.in_channels == 3 and module.kernel_size == (4, 4):
                print(f"  [INFO] Skipping MDEP swap for Swin Patch Partition layer: {name}")
                continue
            
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=(module.bias is not None)
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_swin_linear_with_mdep(module)


def replace_resnet_with_mdep(model):
    r"""Converts Teacher ResNet-18 layers to MDEP structure for state_dict loading.r"""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups, bias=(module.bias is not None)
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features, bias=(module.bias is not None)
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_resnet_with_mdep(module)


# ============================================================================
#  SECTION 5 — SwinMDEPTrainer
# ============================================================================

class SwinMDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs, 
                 teacher_model=None, alpha_d_initial=0.5, alpha_d_final=0.05):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.teacher_model = teacher_model
        
        self.alpha_d_initial = alpha_d_initial
        self.alpha_d_final = alpha_d_final
        
        self.gamma_initial = 5.0
        self.gamma_final = 0.15
        
        self.scaler = torch.amp.GradScaler('cuda')
        self.last_flop_rate = 0.0

    def step_alpha_d(self, epoch):
        if epoch < self.warmup_epochs:
            return self.alpha_d_initial
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        alpha_d = self.alpha_d_final + 0.5 * (self.alpha_d_initial - self.alpha_d_final) * (
            1.0 + math.cos(math.pi * progress)
        )
        return alpha_d

    def step_gamma(self, epoch):
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
        r"""Amortized backward to compute ∂u_a / ∂w and ∂u_e / ∂a^(l).r"""
        self.model.train()

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
        # Class-selective epistemic target to resolve gradient blindness (Eq. 256)
        alpha = uncertainties['alpha']
        u_e_target = torch.mean(torch.sum(1.0 / alpha, dim=1))

        # 1. Microglia: ∂u_a/∂w
        self.model.zero_grad()
        self.reset_effective_weight_grads()
        u_a.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                if m.weight.grad is not None:
                    m.grad_ua_w = m.weight.grad.clone().detach()
                else:
                    m.grad_ua_w = torch.zeros_like(m.weight)

        self.model.zero_grad()
        self.reset_effective_weight_grads()

        # 2. Astrocyte: ∂u_e_target/∂a^(l)
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
                    # Dimensional reduction logic: mean over all dims except channels
                    if isinstance(m, MDEPConv2d):
                        dims = [d for d in range(grad.dim()) if d != 1]
                        m.u_e_node = torch.abs(grad).mean(dim=dims).detach()
                    else:
                        dims = list(range(grad.dim() - 1))
                        m.u_e_node = torch.abs(grad).mean(dim=dims).detach()
                else:
                    m.u_e_node = None

        for h in hooks:
            h.remove()
            
        self.model.zero_grad()
        self.reset_effective_weight_grads()

    def train_epoch(self, epoch, dataloader, device, print_interval=50, scheduler=None):
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
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
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

            inputs, targets = inputs.to(device), targets.to(device)

            # Run morphological updates and amortized gradient computation every N_update = 10 batches (when not in warmup)
            # Or at batch_idx == 0 to initialize
            is_update_step = (batch_idx % 10 == 0)

            if (not is_warmup or epoch < 2) and is_update_step:
                self.compute_amortized_gradients(inputs)

            self.model.zero_grad()
            self.reset_effective_weight_grads()

            with torch.amp.autocast('cuda'):
                student_evidence = self.model(inputs)
                
            distill_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_evidence = self.teacher_model(inputs)
                
                # Evaluate in FP32 to prevent underflow in lgamma/digamma
                student_alpha = student_evidence.float() + 1.0
                teacher_alpha = teacher_evidence.float() + 1.0
                distill_loss = dirichlet_kl_divergence(student_alpha, teacher_alpha)

            with torch.amp.autocast('cuda', enabled=False):
                classification_loss = self.criterion(student_evidence.float(), targets, epoch)
                
            loss = classification_loss + alpha_d * distill_loss
            scaled_loss = loss * current_loss_scale
            
            self.scaler.scale(scaled_loss).backward()

            self.scaler.unscale_(self.optimizer)
            params_to_clip = [p for group in self.optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

            if not is_warmup or epoch < 2:
                inv_scale = 1.0 / (self.scaler.get_scale() + 1e-8)
                for m in self.model.modules():
                    if isinstance(m, (MDEPLinear, MDEPConv2d)):
                        if m.weight.grad is not None:
                            m.grad_L_w = m.weight.grad.clone().detach() * inv_scale
                        else:
                            m.grad_L_w = torch.zeros_like(m.weight)

            self.scaler.step(self.optimizer)
            self.scaler.update()

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
                eta = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)
                
                print(
                    f"    Batch [{batch_idx+1:>4}/{num_batches}] "
                    f"| Loss: {ema_loss:.4f} "
                    f"| Distill: {distill_loss.item():.4f} "
                    f"| Flop: {self.last_flop_rate * 100:.3f}% "
                    f"| ETA: {eta/60:.1f}m",
                    flush=True
                )
                
        return ema_loss


# ============================================================================
#  SECTION 6 — ISIC Dataloaders & Evaluation Suite
# ============================================================================

class ISICDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]['isic_id']
        image = None
        if self.image_dir:
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception:
                    image = None
                    
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        target = self.data_frame.iloc[idx]['target']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)


def get_isic_dataloaders(batch_size=32, debug=False, subsample_ratio=20):
    num_classes = 2
    kaggle_input = '/kaggle/input'
    csv_path = None
    image_dir = None
    
    if os.path.isdir(kaggle_input) and not debug:
        for root, dirs, files in os.walk(kaggle_input):
            if 'train-metadata.csv' in files:
                csv_path = os.path.join(root, 'train-metadata.csv')
                for img_sub in ['train-image/image', 'train-image', 'train-images']:
                    candidate_img = os.path.join(root, img_sub)
                    if os.path.isdir(candidate_img):
                        image_dir = candidate_img
                        break
                break
                
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger for random crop
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if debug or csv_path is None or not os.path.exists(csv_path):
        print("[INFO] Target dataset not found. Falling back to synthetic dataloaders.")
        X = torch.randn(200, 3, 224, 224)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, Y)
        tr = Subset(full, range(140))
        va = Subset(full, range(140, 160))
        te = Subset(full, range(160, 200))
        p_true = [0.5, 0.5]
        p_train = [0.5, 0.5]
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(va, batch_size=batch_size),
                DataLoader(te, batch_size=batch_size),
                num_classes,
                torch.ones(num_classes), p_true, p_train)

    df = pd.read_csv(csv_path)
    # Ensure patient_id exists and has no NaNs
    df = df.dropna(subset=['patient_id']).reset_index(drop=True)
    
    # Group by patient_id and get the max target for each patient to preserve class balance
    patient_df = df.groupby('patient_id')['target'].max().reset_index()
    
    # Split patients into train/test (80/20) stratified by patient-level target
    train_patients, test_patients = train_test_split(
        patient_df, test_size=0.2, stratify=patient_df['target'], random_state=42
    )
    
    # Split train patients into train/val (70/10 of total, which is 12.5% of train)
    train_patients, val_patients = train_test_split(
        train_patients, test_size=0.125, stratify=train_patients['target'], random_state=42
    )
    
    # Map patients back to the original dataframe
    train_df = df[df['patient_id'].isin(train_patients['patient_id'])].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients['patient_id'])].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patients['patient_id'])].reset_index(drop=True)
    
    # Calculate true prior probabilities before subsampling
    class_counts_true = train_df['target'].value_counts().sort_index()
    total_true = len(train_df)
    p_true = [class_counts_true.get(c, 0) / total_true for c in range(num_classes)]

    if subsample_ratio is not None and subsample_ratio > 0:
        train_malignant = train_df[train_df['target'] == 1]
        train_benign = train_df[train_df['target'] == 0]
        num_train_malignant = len(train_malignant)
        if num_train_malignant > 0:
            num_benign_to_sample = min(len(train_benign), num_train_malignant * subsample_ratio)
            train_benign_sampled = train_benign.sample(n=num_benign_to_sample, random_state=42)
            train_df = pd.concat([train_malignant, train_benign_sampled]).reset_index(drop=True)
            print(f"[INFO] Subsampled training set: {num_train_malignant} malignant, {len(train_benign_sampled)} benign.")

    # Calculate train prior probabilities after subsampling
    class_counts_train = train_df['target'].value_counts().sort_index()
    total_train = len(train_df)
    p_train = [class_counts_train.get(c, 0) / total_train for c in range(num_classes)]

    train_ds = ISICDataset(train_df, image_dir, transform=train_tf)
    val_ds = ISICDataset(val_df, image_dir, transform=test_tf)
    test_ds = ISICDataset(test_df, image_dir, transform=test_tf)
    
    import platform
    workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
    total_samples = total_train
    cw_raw = [total_samples / class_counts_train.get(c, 1) for c in range(num_classes)]
    cw = torch.tensor([w / cw_raw[0] for w in cw_raw], dtype=torch.float32)
    return train_loader, val_loader, test_loader, num_classes, cw, p_true, p_train


def compute_ece(confidences, accuracies, n_bins=15):
    r"""Expected Calibration Error with equal-width bins.r"""
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
    r"""Reliability diagram: accuracy vs confidence per bin.r"""
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
    r"""Overlaid histograms of epistemic uncertainty for correct vs wrong.r"""
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
    r"""Precision-Recall Curve with PR-AUC.r"""
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
    r"""Risk-Coverage curve and AURC (Area Under Risk-Coverage).r"""
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


def check_representational_collapse(model):
    r"""
    Diagnoses Representational Collapse in 2:4 structured sparsity.
    r"""
    print("\n[DIAG] Representational Collapse Diagnostics")
    print("-" * 105)
    print(f"  {'Layer':30s} | {'Score Std':12s} | {'Score Mean':12s} | {'Grad Norm':12s} | {'Status'}")
    print("-" * 105)
    
    all_pass = True
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            scores = module.scores.data
            std = scores.std().item() if scores.numel() > 1 else 0.0
            mean = scores.mean().item()
            
            grad_norm = 0.0
            grad_L = getattr(module, 'grad_L_w', None)
            if grad_L is not None:
                grad_norm = grad_L.norm().item()
                
            status = "PASS"
            issues = []
            if std <= 1e-4:
                issues.append("Zero Variance")
            if mean <= -10.0:
                issues.append("Negative Drift")
            if grad_norm <= 1e-6:
                issues.append("Dead Gradient")
                
            if issues:
                status = "FAIL (" + ", ".join(issues) + ")"
                all_pass = False
                
            print(f"  {name:30s} | {std:12.4e} | {mean:12.4f} | {grad_norm:12.4e} | {status}")
            
    print("-" * 105)
    if all_pass:
        print("  OVERALL STATUS: HEALTHY (No Representational Collapse Detected)")
    else:
        print("  WARNING: OVERALL STATUS: WARNING (Representational Collapse Detected in some layers)")
    print()


def print_sparsity_report(model):
    r"""Per-layer and total sparsity stats + 2:4 pattern check + MACs estimation.r"""
    print("\n[SPARSITY] Sparsity & Hardware Metrics Report")
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
            
            macs_dense = n
            macs_sparse = n - z
            total_macs_dense += macs_dense
            total_macs_sparse += macs_sparse
            
            if n % 4 == 0:
                blocks = mask.view(-1, 4)
                valid = (blocks.sum(dim=1) == 2).all().item()
                pattern = "PASS 2:4 (TensorCore Ready)" if valid else "FAIL Not 2:4"
            else:
                pattern = "SKIP (size%4 != 0)"
            print(f"  {name:30s} | {sparsity:5.1f}% sparse | {pattern}")
            
    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0
    print("-" * 75)
    print(f"  {'TOTAL PARAMS':30s} | {overall:5.1f}% sparse")
    print(f"  {'THEORETICAL MACs SAVED':30s} | {macs_saved:5.1f}% reduction in MDEP layers")
    print("  *(Note: Ampere GPU Tensor Cores provide 2x speedup for strict 2:4 sparsity)*")
    print()
    
    check_representational_collapse(model)


@torch.no_grad()
def evaluate(model, val_loader, test_loader, device, num_classes, plot=True):
    r"""Full evaluation: metrics, plots, and uncertainty analysis.r"""
    model.eval()

    best_t = 0.5
    if num_classes == 2:
        val_targets = []
        val_probs = []
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            evidence = model(inputs)
            unc = compute_uncertainties(evidence)
            p_hat = (unc['alpha'] / unc['S']).cpu().numpy()
            val_targets.append(targets.numpy())
            val_probs.append(p_hat)
            
        val_y_true = np.concatenate(val_targets)
        val_probs = np.concatenate(val_probs, axis=0)
        
        best_bal_acc = 0.0
        for t in np.linspace(0.01, 0.99, 199):
            y_pred_t = (val_probs[:, 1] >= t).astype(int)
            bal_acc_t = balanced_accuracy_score(val_y_true, y_pred_t)
            if bal_acc_t > best_bal_acc:
                best_bal_acc = bal_acc_t
                best_t = t
        print(f"[Validation] Optimized decision threshold: {best_t:.4f}")

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
        preds = p_hat.argmax(axis=1) if num_classes > 2 else (p_hat[:, 1] >= best_t).astype(int)
        confs = p_hat.max(axis=1)

        all_targets.append(targets.numpy())
        all_preds.append(preds)
        all_confs.append(confs)
        all_probs.append(p_hat)
        all_u_e.append(unc['epistemic'].cpu().numpy()[:, 0])
        all_u_a.append(unc['aleatoric'].cpu().numpy()[:, 0])

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    confs  = np.concatenate(all_confs)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    correct = (y_pred == y_true).astype(float)

    # ── Scalar Metrics ───────────────────
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
    print("\n[EVAL] Evaluation Results (Threshold = 0.5)")
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
        print(f"\n[EVAL] Evaluation Results (Optimized Threshold = {best_t:.4f})")
        print("=" * 50)
        print(f"  Balanced Accuracy     : {bal_acc_opt:.4f}")
        print(f"  Macro F1-Score        : {macro_f1_opt:.4f}")
        print(f"  Sensitivity (Recall)  : {sensitivity_opt:.4f}")
        print(f"  Specificity           : {specificity_opt:.4f}")
        print("=" * 50)

    # ── Plots ──────────────────────────────────────────────────────
    if plot:
        plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
        plot_uncertainty_histogram(
            u_e[correct.astype(bool)],
            u_e[~correct.astype(bool)],
        )
        if num_classes == 2:
            plot_pr_curve(y_true, probs[:, 1])
        plot_risk_coverage_curve(y_true, y_pred_opt, confs)

    metrics = {
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'macro_auroc': macro_auroc,
        'pauc': pauc,
        'ece': ece_val,
        'minority_ece': m_ece,
        'best_threshold': best_t,
        'bal_acc_opt': bal_acc_opt,
        'macro_f1_opt': macro_f1_opt,
        'sensitivity_opt': sensitivity_opt,
        'specificity_opt': specificity_opt,
        'mean_epistemic': u_e.mean(),
        'mean_aleatoric': u_a.mean()
    }

    return macro_auroc, metrics


def load_state_dict_filtered(model, state_dict, strict=False):
    r"""Loads state_dict into model, filtering out non-persistent mask/tau buffers for compatibility.r"""
    for key in list(state_dict.keys()):
        if key.endswith('.mask') or key.endswith('.tau'):
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=strict)


def load_resnet_teacher(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Teacher checkpoint file '{checkpoint_path}' not found. Distillation disabled.")
        return None
        
    print(f"[INFO] Rebuilding ResNet-18 Teacher from checkpoint: {checkpoint_path}")
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2),
            EvidenceLayer(activation='softplus')
        )
        replace_resnet_with_mdep(model)
        
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        load_state_dict_filtered(model, state_dict, strict=False)
        model.to(device)
        model.eval()
        
        for p in model.parameters():
            p.requires_grad = False
            
        print("ResNet Teacher loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load ResNet Teacher: {e}")
        return None


# ============================================================================
#  SECTION 7 — main() entrypoint
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Set parameters with argparse support
    import argparse
    is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
    
    if not is_interactive:
        parser = argparse.ArgumentParser(description="Swin-MDEP Training Script")
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--warmup-epochs", type=int, default=10)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--subsample", type=int, default=20)
        parser.add_argument("--wandb-project", type=str, default="swin-mdep")
        parser.add_argument("--wandb-run-id", type=str, default=None)
        parser.add_argument("--resume", action="store_true")
        args, _ = parser.parse_known_args()
    else:
        class Args:
            epochs = 30
            warmup_epochs = 10
            batch_size = 32
            debug = False
            subsample = 20
            wandb_project = "swin-mdep"
            wandb_run_id = None
            resume = False
        args = Args()

    epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    batch_size = args.batch_size
    debug_mode = args.debug
    subsample_ratio = args.subsample

    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="must" if (args.resume and args.wandb_run_id) else "allow",
            config={
                "epochs": epochs,
                "warmup_epochs": warmup_epochs,
                "batch_size": batch_size,
                "learning_rate": 2.0e-5,
                "weight_decay": 0.01,
                "subsample_ratio": subsample_ratio,
                "debug_mode": debug_mode
            }
        )
    except Exception as e:
        print(f"[WARNING] Failed to initialize Weights & Biases: {e}")
        print("[INFO] Falling back to disabled mode (console-only logging).")
        try:
            wandb.init(project=args.wandb_project, mode="disabled")
        except Exception:
            class DummyRun:
                def __init__(self):
                    self.id = "dummy"
                    self.config = {}
                def log(self, *args, **kwargs): pass
                def finish(self, *args, **kwargs): pass
                def log_artifact(self, *args, **kwargs): pass
            wandb.run = DummyRun()

    # Check if dataset is present locally, if not fall back to synthetic
    train_loader, val_loader, test_loader, num_classes, class_weights, p_true, p_train = get_isic_dataloaders(
        batch_size=batch_size, debug=debug_mode, subsample_ratio=subsample_ratio
    )

    print("[INFO] Initializing Swin-T Student model...")
    try:
        student = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    except Exception:
        print("[WARNING] Could not load Swin_T ImageNet weights. Initializing randomly.")
        student = models.swin_t(weights=None)
        
    # Replace head with EDL
    in_features = student.head.in_features
    student.head = nn.Sequential(
        nn.Linear(in_features, num_classes),
        LogPriorCorrection(p_true, p_train),
        EvidenceLayer(activation='softplus')
    )
    nn.init.normal_(student.head[0].weight, mean=0, std=0.001)
    nn.init.constant_(student.head[0].bias, 0)
    
    # Swap inner linear layers
    replace_swin_linear_with_mdep(student.features)
    student.to(device)

    # Load ResNet Teacher checkpoint if it exists in the workspace
    candidate_paths = [
        'model_checkpoint.pth',
        '../RESNET50 backbone/model_checkpoint.pth',
        './RESNET50 backbone/model_checkpoint.pth',
        '/kaggle/input/mdep-resnet50-backbone/model_checkpoint.pth',
    ]
    teacher_path = 'model_checkpoint.pth'
    found_teacher = False
    
    # Check candidates
    for path in candidate_paths:
        if os.path.exists(path):
            teacher_path = path
            found_teacher = True
            break
            
    # Fallback: recursive search in '/kaggle/input' or current folder
    if not found_teacher:
        search_dirs = ['/kaggle/input', '.']
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                for root, dirs, files in os.walk(s_dir):
                    if 'model_checkpoint.pth' in files:
                        teacher_path = os.path.join(root, 'model_checkpoint.pth')
                        found_teacher = True
                        break
            if found_teacher:
                break
                
    teacher = load_resnet_teacher(device, teacher_path)

    criterion = EvidentialFocalLoss(
        gamma=2.0, num_classes=num_classes, kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=warmup_epochs, total_epochs=epochs
    )

    trainable_params = [p for name, p in student.named_parameters() if 'scores' not in name]
    optimizer = optim.AdamW(trainable_params, lr=2.0e-5, weight_decay=0.01)

    # Cosine annealing scheduler (warmup is handled in train_epoch for epoch < 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - 1, eta_min=1.0e-6
    )

    trainer = SwinMDEPTrainer(
        model=student,
        optimizer=optimizer,
        criterion=criterion,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        teacher_model=teacher
    )

    save_path = 'swin_mdep_checkpoint.pth'
    best_save_path = 'swin_mdep_best_checkpoint.pth'
    start_epoch = 0
    best_macro_auroc = 0.0

    # Resume logic
    if args.resume:
        print("[INFO] Attempting to resume training...")
        checkpoint_path = save_path
        
        # Download checkpoint from WandB Artifacts if run ID is provided
        if args.wandb_run_id:
            try:
                print(f"Downloading checkpoint artifact: swin-mdep-checkpoint-{args.wandb_run_id}:latest")
                artifact = wandb.use_artifact(f"swin-mdep-checkpoint-{args.wandb_run_id}:latest")
                artifact_dir = artifact.download()
                checkpoint_path = os.path.join(artifact_dir, save_path)
            except Exception as e:
                print(f"[WARNING] Could not download WandB checkpoint artifact: {e}. Falling back to local search.")
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                load_state_dict_filtered(student, checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_macro_auroc = checkpoint.get('best_macro_auroc', 0.0)
                print(f"[INFO] Resumed training from epoch {start_epoch} with best Macro-AUROC {best_macro_auroc:.4f}")
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint: {e}")
                sys.exit(1)
        else:
            print(f"[WARNING] Checkpoint file '{checkpoint_path}' not found. Starting training from epoch 0.")

    print("\nStarting Swin-T MDEP Training Run")
    print("=" * 80)
    for epoch in range(start_epoch, epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, scheduler=scheduler)
        if epoch >= 1:  # LR warmup period is 1 epoch, after that scheduler takes over
            scheduler.step()
        phase = "Warm-up (Dense)" if epoch < warmup_epochs else "MDEP 2:4 Active Sparsity"
        gamma = trainer.step_gamma(epoch)
        alpha_d = trainer.step_alpha_d(epoch)
        
        print(
            f"  Epoch [{epoch+1:>2}/{epochs:>2}] "
            f"| Loss: {loss:.4f} "
            f"| Phase: {phase:<24} "
            f"| gamma: {gamma:.3f} | alpha_d: {alpha_d:.3f}"
        )
        
        # Log training stats
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "gamma": gamma,
            "alpha_d": alpha_d,
            "phase_warmup": int(epoch < warmup_epochs)
        }, step=epoch)

        # Intermediate evaluation (validation)
        val_macro_auroc, val_metrics = evaluate(student, val_loader, val_loader, device, num_classes, plot=False)
        val_logs = {f"val_{k}": v for k, v in val_metrics.items()}
        val_logs["epoch"] = epoch + 1
        wandb.log(val_logs, step=epoch)

        is_best = val_macro_auroc > best_macro_auroc
        if is_best:
            best_macro_auroc = val_macro_auroc
            print(f"New best validation Macro-AUROC: {best_macro_auroc:.4f}")

        # Save checkpoint dict
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': trainer.scaler.state_dict(),
            'best_macro_auroc': best_macro_auroc
        }
        
        # Save current checkpoint locally
        torch.save(checkpoint_dict, save_path)
        print(f"Checkpoint saved locally to '{save_path}'")
        
        # Log checkpoint to wandb
        try:
            run_id = wandb.run.id if (wandb.run and hasattr(wandb.run, 'id') and wandb.run.id) else "offline"
            artifact = wandb.Artifact(name=f"swin-mdep-checkpoint-{run_id}", type="model")
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[WARNING] Failed to log WandB checkpoint artifact: {e}")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint_dict, best_save_path)
            print(f"Best checkpoint saved locally to '{best_save_path}'")
            try:
                run_id = wandb.run.id if (wandb.run and hasattr(wandb.run, 'id') and wandb.run.id) else "offline"
                best_artifact = wandb.Artifact(name=f"swin-mdep-best-checkpoint-{run_id}", type="model")
                best_artifact.add_file(best_save_path)
                wandb.log_artifact(best_artifact)
            except Exception as e:
                print(f"[WARNING] Failed to log best WandB checkpoint artifact: {e}")

    print("=" * 80)
    print("Training sequence completed.")

    # Load best checkpoint for final evaluation plots
    if os.path.exists(best_save_path):
        print(f"[INFO] Loading best model checkpoint from '{best_save_path}' for final evaluation plots...")
        checkpoint = torch.load(best_save_path, map_location=device, weights_only=False)
        load_state_dict_filtered(student, checkpoint['model_state_dict'], strict=False)

    evaluate(student, val_loader, test_loader, device, num_classes, plot=True)
    print_sparsity_report(student)
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
