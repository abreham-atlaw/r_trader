import typing

import torch.nn as nn


class LinearModel(nn.Module):
	def __init__(
			self,
			block_size: int,
			vocab_size: int,
			layer_sizes: typing.List[int],
			dropout_rate: float = 0,
			activation: typing.Optional[nn.Module] = None,
			init_fn: typing.Optional[typing.Callable] = None,
			norm: typing.Union[bool, typing.List[bool]] = False
	):
		super(LinearModel, self).__init__()
		layer_sizes = [block_size] + layer_sizes + [vocab_size]

		if isinstance(norm, bool):
			norm = [norm for _ in range(len(layer_sizes))]
		if len(norm) != len(layer_sizes)-1:
			raise ValueError("Norm size doesn't match layers size")
		self.layers = nn.ModuleList()

		for i in range(len(layer_sizes) - 1):
			if norm[i]:
				self.layers.append(nn.LayerNorm(layer_sizes[i]))
			self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			if init_fn is not None:
				init_fn(self.layers[-1].weight)
		if activation is None:
			activation = nn.Identity()
		self.activation = activation

		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
			x = self.activation(x)
			x = self.dropout(x)
		return x
