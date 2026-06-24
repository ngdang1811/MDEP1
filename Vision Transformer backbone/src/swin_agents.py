import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothedSTE(torch.autograd.Function):
    """
    Smoothed Straight-Through Estimator with Local 2:4 Bounds.
    Forward: passes the hard binary mask unchanged, but utilizes precomputed local thresholds.
    Backward: approximates dM/dS ≈ sigma'((S - tau)/gamma) so gradients flow
              only to connections near the 2:4 survival boundary.
    """
    @staticmethod
    def forward(ctx, scores, mask, tau, gamma):
        ctx.save_for_backward(scores, tau, torch.tensor(gamma))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, tau, gamma = ctx.saved_tensors
        gamma_val = gamma.item()
        
        # Localized Leaky STE: margin to the boundary
        margin = scores - tau
        
        sig = torch.sigmoid(margin / gamma_val)
        # Added leaky term (0.05) to ensure non-vanishing gradient flow for non-boundary scores
        grad_scores = grad_output * (sig * (1.0 - sig) / gamma_val + 0.05)
        return grad_scores, None, None, None

def generate_2_4_mask(scores):
    """
    Generates a 2:4 structured sparsity mask dynamically.
    For every contiguous block of 4 elements, the top 2 elements (by score) survive.
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
        # Initialize latent scores to weight magnitude to prevent pruning shock
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('tau', torch.zeros_like(self.weight), persistent=False)
        self.register_buffer('scores_momentum', torch.zeros_like(self.weight))
        self.gamma = 1.0  # STE temperature
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.tau, self.gamma)
            effective_weight = self.weight * differentiable_mask
            
        if effective_weight.requires_grad and not effective_weight.is_leaf:
            effective_weight.retain_grad()
        # Bypass PyTorch module attribute assignment to prevent unwanted parameter registration
        self.__dict__['effective_weight'] = effective_weight
            
        return F.linear(x, effective_weight, self.bias)

class MDEPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MDEPConv2d, self).__init__(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.scores = nn.Parameter(torch.abs(self.weight.data).clone())
        self.register_buffer('mask', torch.ones_like(self.weight))
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
    """
    Updates latent scores S_ij using Microglia (pruning) and Astrocyte (growing) signals.
    Employs the corrected opposing forces: delta_S = G_ij - C_ij.
    Computes and caches new 2:4 mask and tau configurations once.
    """
    total_flips = 0
    total_elements = 0
    
    print("\n[MDEP Multi-Agent Structure Update]")
    print("-" * 80)
    
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                continue
            
            # Record previous mask configuration from the cache
            old_mask = module.mask.clone()
            w_val = module.weight.data
            
            # --- 1. Microglia agent: Pruning Signal (C_ij) ---
            # c1: Class-predictive importance = |w * ∂L_EFL/∂w|
            c1 = torch.abs(w_val * module.grad_L_w)
            c1_norm = torch.tanh(c1 / (c1.median() + 1e-8))
            
            # c2: Aleatoric noise-modeling importance = |w * ∂u_a/∂w|
            grad_ua_w = getattr(module, 'grad_ua_w', torch.zeros_like(w_val))
            c2 = torch.abs(w_val * grad_ua_w)
            c2_norm = torch.tanh(c2 / (c2.median() + 1e-8))
            
            # Microglia driving force (inhibits connections) using robust Tanh scaling
            C_ij = c1_norm + beta * c2_norm
            
            # --- 2. Astrocyte agent: Growing Signal (G_ij) ---
            # g1: Epistemic uncertainty gradient w.r.t activation nodes
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
            
            # g2: Classification gradient magnitude = |∂L_EFL/∂w|
            g2 = torch.abs(module.grad_L_w)
            g2_norm = torch.tanh(g2 / (g2.mean() + 1e-8))
            
            # Astrocyte driving force (encourages growth) using robust Tanh scaling
            G_ij = g1_norm * g2_norm
            
            # Anti-crystallization logic: inject small noise if growth signals are frozen to 0
            if G_ij.max().item() <= 1e-8:
                noise = 0.0316 * torch.randn_like(G_ij) * g1_norm
                G_ij = G_ij + torch.clamp(noise, min=0.0)
                
            # --- 3. Dynamic Opposing Forces (Delta S_ij) ---
            # Corrected formulation: Growth promotes connection (+), Pruning demotes it (-)
            delta_S = G_ij - C_ij
            
            # Step 1: Update structural velocity (EMA momentum)
            beta_m = 0.95
            module.scores_momentum.data.mul_(beta_m).add_(delta_S, alpha=1.0 - beta_m)
            
            # Step 2: Update latent scores
            eta = 0.02
            module.scores.data.add_(module.scores_momentum.data, alpha=eta)
            
            # Step 3: Zero-centering stabilization to prevent parameter drift
            module.scores.data.sub_(module.scores.data.mean())
            
            # Step 4: Clamp to prevent infinite growth and saturation
            module.scores.data.clamp_(min=-5.0, max=5.0)
            
            # Step 5: Update the cached 2:4 structured mask and local thresholds (tau)
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
            
            # Count mask updates (flips) for convergence telemetry
            flips = (old_mask != new_mask).sum().item()
            total_flips += flips
            total_elements += old_mask.numel()
            
            print(f"  Layer: {name:<40} | Flips: {flips:>5} / {old_mask.numel():<7} ({flips / old_mask.numel() * 100:.4f}%)")
            
    flop_rate = total_flips / (total_elements + 1e-8)
    print(f"  >>> TOTAL MASK FLOP RATE: {flop_rate * 100:.6f}% ({total_flips} / {total_elements})")
    print("-" * 80)
    return flop_rate
