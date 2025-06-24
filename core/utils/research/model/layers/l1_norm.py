import torch
from torch import nn


class L1Norm(nn.Module):

	def __init__(self, *args, n: float = 1, dim=-1, **kwargs):
		super().__init__(*args, **kwargs)
		self.n = n
		self.dim = dim

	def forward(self, x: torch.Tensor):
		return x/torch.sum(x, keepdim=True, dim=self.dim)
