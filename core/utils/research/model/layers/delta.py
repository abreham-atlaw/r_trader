import torch
import torch.nn as nn


class Delta(nn.Module):

    def __init__(self, dim=-1, n: int = 1):
        super().__init__()
        self.dim = dim
        self.n = n

    def forward(self, inputs):
        current = torch.narrow(inputs, self.dim, self.n, inputs.size(self.dim) - self.n)
        previous = torch.narrow(inputs, self.dim, 0, inputs.size(self.dim) - self.n)
        return current - previous
