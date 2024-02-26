import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, size, softmax=False, device='cpu', a=0.1):
        super(WeightedMSELoss, self).__init__()
        self.size = size
        self.w = (1/a) ** (size - torch.abs(torch.arange(size).unsqueeze(0) - torch.arange(size).unsqueeze(1))) * a**size
        self.w = self.w.to(device)  # move w to the correct device
        self.softmax = softmax

    def forward(self, y_hat, y):
        if self.softmax:
            y_hat = F.softmax(y_hat, dim=1)
        return torch.mul(((y.unsqueeze(1)-y_hat.unsqueeze(2))**2)*y.unsqueeze(1), self.w).mean()
