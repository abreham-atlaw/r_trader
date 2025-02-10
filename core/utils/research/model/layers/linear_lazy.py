import torch
from torch import nn


class LinearLazy(nn.Module):

	def __init__(self, out_features: int, *args, **kwargs):
		super().__init__()
		self.out_features = out_features
		self.layer = None
		self.args, self.kwargs = args, kwargs

	def forward(self, x: torch.Tensor):
		if self.layer is None:
			self.layer = nn.Linear(x.shape[-1], self.out_features, *self.args, **self.kwargs)
		return self.layer(x)
