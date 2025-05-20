import typing

import torch
from torch import nn


class DynamicPool(nn.Module):

	def __init__(self, pool_range: typing.Tuple[float, float], pool_size: int, stride: int = 1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.pool_range = pool_range
		self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=stride)

	def forward(self, x: torch.Tensor) -> torch.Tensor:

		true_range = tuple([int(self.pool_range[i] * x.shape[-1]) for i in range(2)])
		return torch.concat([
			x[tuple([slice(None)] * (x.ndim - 1) + [slice(0, true_range[0])])],
			self.pool(x[tuple([slice(None)] * (x.ndim - 1) + [slice(true_range[0], true_range[1])])]),
			x[tuple([slice(None)] * (x.ndim - 1) + [slice(true_range[1], x.shape[-1])])]
		], dim=-1)
