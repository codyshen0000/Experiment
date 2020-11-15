import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=7, beta=3.7, **kwargs):
        super(FocalLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, y):
        L1 = torch.abs(x - y)
        out = L1 / (1 + torch.exp(self.alpha * (self.beta - L1)))
        return out.mean()

  


