import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(alpha, num_classes):
    """
    Kullback-Leibler divergence between the Dirichlet distribution
    parameterized by alpha and a uniform Dirichlet distribution.
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
        """
        Args:
            evidence (torch.Tensor): Evidence (batch_size, num_classes)
            targets (torch.Tensor): One-hot targets or class indices (batch_size, ...)
            epoch (int, optional): Current training epoch for KL annealing & Focal Gamma scheduling
        """
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()
            
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Expected probability
        p_hat = alpha / S
        
        # Cross entropy term: sum_c y_c * (digamma(S) - digamma(alpha_c))
        loss_ce = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
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
        
        # Focal modulation
        p_target = torch.sum(targets * p_hat, dim=1, keepdim=True)
        focal_weight = (1.0 - p_target.detach()) ** gamma_val
        
        if self.class_weights is not None:
            sample_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
        else:
            sample_weight = 1.0
            
        # KL Divergence Regularization
        alpha_tilde = targets + (1 - targets) * alpha
        loss_kl = kl_divergence(alpha_tilde, self.num_classes)
        
        # KL Annealing
        if epoch is not None and self.annealing_epochs > 0:
            annealing_coef = min(1.0, epoch / self.annealing_epochs)
        else:
            annealing_coef = 1.0
            
        # Final modulated loss (scale the CE loss by focal weight, and the overall loss by sample weight to balance KL and CE forces under class imbalance)
        loss = sample_weight * (focal_weight * loss_ce + self.kl_lambda * annealing_coef * loss_kl)
        
        return torch.mean(loss)
