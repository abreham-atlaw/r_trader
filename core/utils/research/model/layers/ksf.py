import torch
import torch.nn as nn


class KalmanStaticFilter(nn.Module):
    def __init__(self, alpha: float, beta: float, *args, **kwargs):
        self.a = alpha
        self.b = beta
        super(KalmanStaticFilter, self).__init__(*args, **kwargs)

    def forward(self, Z, *args, **kwargs):
        X = torch.zeros_like(Z[:, 0:0])
        p = Z[:, 0]
        v = torch.zeros_like(p)

        for i in range(Z.shape[1]):
            diff = Z[:, i] - p
            X = torch.cat(
                (
                    X,
                    (p + (self.a * diff)).unsqueeze(-1)
                ), dim=1)
            v = v + self.b * diff
            p = X[:, i] + v

        return X
