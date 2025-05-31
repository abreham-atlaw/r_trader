import torch
import torch.nn as nn


class FlattenLayer(nn.Module):

	def __init__(self, start_dim: int, end_dim: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.start_dim, self.end_dim = start_dim, end_dim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.flatten(x, self.start_dim, self.end_dim)
