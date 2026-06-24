import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_agents import MDEPLinear, MDEPConv2d

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
    Computes epistemic and aleatoric uncertainties from the Dirichlet evidence.
    
    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)
        
    Returns:
        dict: Epistemic uncertainty, aleatoric uncertainty, alpha, and Dirichlet strength S.
    """
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

import torchvision.models.swin_transformer as swin_utils
import types

def mdep_swa_forward(self, x):
    relative_position_bias = self.get_relative_position_bias()
    
    def get_effective_weight(module):
        if hasattr(module, 'warmup'):
            if getattr(module, 'warmup', True):
                effective_weight = module.weight
            else:
                from swin_agents import SmoothedSTE
                differentiable_mask = SmoothedSTE.apply(module.scores, module.mask, module.tau, module.gamma)
                effective_weight = module.weight * differentiable_mask
            if effective_weight.requires_grad and not effective_weight.is_leaf:
                effective_weight.retain_grad()
            module.__dict__['effective_weight'] = effective_weight
            return effective_weight
        return module.weight

    qkv_weight = get_effective_weight(self.qkv)
    proj_weight = get_effective_weight(self.proj)
    
    return swin_utils.shifted_window_attention(
        x,
        qkv_weight,
        proj_weight,
        relative_position_bias,
        self.window_size,
        self.num_heads,
        shift_size=self.shift_size,
        attention_dropout=self.attention_dropout,
        dropout=self.dropout,
        qkv_bias=self.qkv.bias,
        proj_bias=self.proj.bias,
        training=self.training,
    )

def replace_swin_linear_with_mdep(model):
    """
    Recursively swaps nn.Linear layers inside Swin Transformer features to MDEPLinear.
    Copies pretrained weights and initializes latent scores to weight magnitudes.
    """
    # Patch all ShiftedWindowAttention instances to use effective_weight
    # since PyTorch's native implementation bypasses the qkv/proj forward methods.
    if getattr(model, '_mdep_swa_patched', None) is None:
        for name, module in model.named_modules():
            if isinstance(module, swin_utils.ShiftedWindowAttention):
                module.forward = types.MethodType(mdep_swa_forward, module)
        model._mdep_swa_patched = True

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Check if this is indeed a target layer (e.g. not NonDynamicallyQuantizableLinear of MHA if ViT, 
            # but for Swin it's standard nn.Linear inside attention and MLP)
            new = MDEPLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None)
            )
            # Copy weights and biases
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Conv2d):
            # We skip features[0][0] which is the Patch Partition layer to preserve low-level visual stems
            # Check module parameters to decide. Features[0][0] is Conv2d(3, 96, 4, 4)
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
