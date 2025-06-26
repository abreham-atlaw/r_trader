import torch
from torch import nn

from .add import Add


class AddAndNorm(nn.Module):

	def __init__(self, norm_layer, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.norm = norm_layer
		self.add = Add()

	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return self.norm(self.add(x, y))

