import torch
import torch.nn as nn

class Delta(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        current = torch.narrow(inputs, self.dim, 1, inputs.size(self.dim) - 1)
        previous = torch.narrow(inputs, self.dim, 0, inputs.size(self.dim) - 1)
        return current - previous
