import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.__cross_entropy = CrossEntropyLoss()

    def forward(self, y_true, y_pred):
        diff = torch.abs(torch.argmax(y_true, dim=1) - torch.argmax(y_pred, dim=1))
        weights = diff.float() ** 2
        ce_loss = self.__cross_entropy(y_pred, y_true)
        weighted_loss = weights * ce_loss
        return weighted_loss.mean()
        return ce_loss

