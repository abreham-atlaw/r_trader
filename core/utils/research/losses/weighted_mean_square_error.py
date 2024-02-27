import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, size, softmax=True, device='cpu', a=0.15):
        super(WeightedMSELoss, self).__init__()
        self.size = size
        self.w = self.__initialize_weights(size, a)
        self.w = self.w.to(device)
        self.softmax = softmax

    @staticmethod
    def __initialize_weights(size, a):
        x = ((size - torch.abs(torch.arange(size).unsqueeze(0) - torch.arange(size).unsqueeze(1)))/size)
        return torch.exp(-(((1 - x) ** 2) / (a)))

    def forward(self, y_hat, y):
        if self.softmax:
            y_hat = F.softmax(y_hat, dim=1)
        return torch.mul(((y.unsqueeze(1)-y_hat.unsqueeze(2))**2)*y.unsqueeze(1), self.w).mean()
