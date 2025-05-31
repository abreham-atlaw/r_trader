import torch
import torch.nn as nn


class DynamicBatchNorm(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.norm_layer = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.norm_layer is None:
			self.norm_layer = nn.BatchNorm1d(num_features=x.shape[-1])
		if x.shape[0] == 1:
			return x
		return self.norm_layer(x)
