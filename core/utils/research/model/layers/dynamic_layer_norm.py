import torch
from torch import nn


class DynamicLayerNorm(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.norm_layer = None

	def norm(self, x: torch.Tensor):
		if self.norm_layer is None:
			self.norm_layer = nn.LayerNorm(x.size()[1:])
		return self.norm_layer(x)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.norm(x)
