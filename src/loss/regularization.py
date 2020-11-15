import torch
import torch.nn as nn
import torch.nn.functional as F


class Lp_Regular(nn.Module):
    def __init__(self, p=0.1, **kwargs):
        super(Lp_Regular, self).__init__()
        self.p = p
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def forward(self, model):
        RW_loss = 0.0
        for param in model.parameters():
            RW_loss += torch.sum(torch.pow(torch.abs(param),0.7))
        return RW_loss.mean()