import typing
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class LayerConfig:
	unit: int
	dropout: float = 0
	activation: typing.Optional[nn.Module] = None
	norm: bool = False


class LinearModel(nn.Module):
	def __init__(
			self,
			block_size: int,
			vocab_size: int,
			layers: typing.List[LayerConfig],
			init_fn: typing.Optional[typing.Callable] = None,
	):
		super(LinearModel, self).__init__()

		layers = [
			LayerConfig(
				block_size
			)
		] + layers + [
			LayerConfig(
				vocab_size
			)
		]

		self.layers = nn.ModuleList()

		for i, p_layer_config in enumerate(layers[:-1]):
			layer_config = layers[i+1]
			if layer_config.norm:
				self.layers.append(nn.LayerNorm(p_layer_config.unit))
			self.layers.append(nn.Linear(p_layer_config.unit, layer_config.unit))
			if init_fn is not None:
				init_fn(self.layers[-1].weight)
			if layer_config.activation is not None:
				self.layers.append(layer_config.activation)
			if layer_config.dropout not in [None, 0]:
				self.layers.append(nn.Dropout(layer_config.dropout))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
