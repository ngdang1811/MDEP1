import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidenceLayer(nn.Module):
    """
    Ensures the output of the network is non-negative evidence (e >= 0).
    Typically replaces the final Softmax layer.
    """
    def __init__(self, activation='softplus'):
        super(EvidenceLayer, self).__init__()
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x):
        return self.activation(x)

def compute_uncertainties(evidence):
    """
    Computes epistemic and aleatoric uncertainties from the evidence.
    
    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)
        
    Returns:
        dict: A dictionary containing epistemic (u_e) and aleatoric (u_a) uncertainties.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = evidence.shape[1]
    
    # Epistemic Uncertainty: K / S
    u_e = K / S
    
    # Aleatoric Uncertainty: - sum (alpha_c / S) * (digamma(S + 1) - digamma(alpha_c + 1))
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
        'S': S
    }
