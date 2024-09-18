import torch
from torch import nn


class MinMaxNorm(nn.Module):

	def __init__(self, *args, dim=-1, **kwargs):
		super().__init__(*args, **kwargs)
		self.dim = dim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		min_value = torch.min(x, dim=self.dim, keepdim=True).values
		x = (x - min_value)
		return x/torch.max(x, dim=self.dim, keepdim=True).values
