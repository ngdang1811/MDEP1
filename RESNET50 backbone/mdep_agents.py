import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Generates a 2:4 structured sparsity mask dynamically.
    For every block of 4 elements, the top 2 elements (by score) are kept.
    This effectively uses a dynamic threshold tau_t locally.
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
    def __init__(self, in_features, out_features, bias=True):
        super(MDEPLinear, self).__init__(in_features, out_features, bias)
        self.scores = nn.Parameter(torch.randn_like(self.weight))
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0 # Temperature for SmoothedSTE
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(MDEPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.scores = nn.Parameter(torch.randn_like(self.weight))
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
            
        return F.conv2d(x, effective_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def update_scores_agents(model, beta=1.0):
    """
    Updates latent scores S_ij using Microglia (pruning) and Astrocyte (growing) signals (Dense Gradient Pass).
    """
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                continue
            
            w_val = module.weight.data
            
            # --- Microglia agent: pruning score (§5.2) ---
            # c1: importance for prediction = |w * ∂L_EFL/∂w|
            c1 = torch.abs(w_val * module.grad_L_w)
            c1_min = c1.min()
            c1_max = c1.max()
            c1_norm = (c1 - c1_min) / (c1_max - c1_min + 1e-8)
            
            # c2: importance for noise modelling = |w * ∂u_a/∂w|
            grad_ua_w = getattr(module, 'grad_ua_w', torch.zeros_like(w_val))
            c2 = torch.abs(w_val * grad_ua_w)
            c2_min = c2.min()
            c2_max = c2.max()
            c2_norm = (c2 - c2_min) / (c2_max - c2_min + 1e-8)
            
            C_ij = c1_norm + beta * c2_norm
            
            # --- Astrocyte agent: growing score (§5.3) ---
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
            g1_min = g1.min()
            g1_max = g1.max()
            g1_norm = (g1 - g1_min) / (g1_max - g1_min + 1e-8)
            
            # g2: per-weight loss gradient magnitude = |∂L_EFL/∂w|
            g2 = torch.abs(module.grad_L_w)
            g2_min = g2.min()
            g2_max = g2.max()
            g2_norm = (g2 - g2_min) / (g2_max - g2_min + 1e-8)
            
            G_ij = g1_norm * g2_norm
            
            # Khắc phục hiện tượng kết tinh Astrocyte (Phase 4 Action 2):
            # Nếu tất cả các tiềm năng tăng trưởng G_ij bằng 0, thêm xung lực tăng trưởng ngẫu nhiên
            if G_ij.max().item() <= 1e-8:
                noise = 0.0316 * torch.randn_like(G_ij) * g1_norm
                G_ij = G_ij + torch.clamp(noise, min=0.0)
            
            # Corrected formulation: Growth promotes connection (+), Pruning demotes it (-)
            delta_S = G_ij - C_ij
            
            # Step 1: Update Velocity (Momentum EMA)
            beta_m = 0.95
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            
            # Step 2: Update Latent Scores S
            eta = 0.02
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            
            # Step 3: Zero-Centering Stabilization to prevent memory drift
            module.scores.data.sub_(module.scores.data.mean())
            
            # Step 4: Clamp scores to prevent infinite growth and gradient underflow
            module.scores.data.clamp_(min=-5.0, max=5.0)
