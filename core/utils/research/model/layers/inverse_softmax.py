import torch
import torch.nn as nn


class InverseSoftmax(nn.Module):

	def __init__(self, *args, eps: float = 1e-6, **kwargs):
		super().__init__(*args, **kwargs)
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.log(x + self.eps)
