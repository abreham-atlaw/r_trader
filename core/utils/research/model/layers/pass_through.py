import torch
import torch.nn as nn


class PassThroughLayer(nn.Module):

	def __init__(self, layer, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.layer = layer

	def forward(self, *args, **kwargs) -> torch.Tensor:
		return self.layer(*args, **kwargs)
