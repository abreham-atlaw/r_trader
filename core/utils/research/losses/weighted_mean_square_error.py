import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, size, softmax=True, device='cpu', a=0.15):
        super(WeightedMSELoss, self).__init__()
        self.size = size
        self.w = self.__init_weight_matrix(size, a)
        self.w = self.w.to(device)
        self.softmax = softmax

    @staticmethod
    def __init_weight_matrix(size, a):
        x = ((size - torch.abs(torch.arange(size).unsqueeze(0) - torch.arange(size).unsqueeze(1)))/size)
        return torch.exp(-(((1 - x) ** 2) / (a)))

    def forward(self, y_hat, y):
        if self.softmax:
            y_hat = F.softmax(y_hat, dim=1)

        mse = ((y.unsqueeze(1)-y_hat.unsqueeze(2))**2)
        unweighted = mse * y.unsqueeze(1)
        weighted = torch.mul(unweighted, self.w)
        return torch.mean(weighted)
