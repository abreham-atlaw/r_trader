import importlib
import typing

import torch.nn as nn

from core.utils.research.model.model.savable import SavableModel


class LinearModel(SavableModel):

	def __init__(
			self,
			block_size: int,
			vocab_size: int,
			layer_sizes: typing.List[int],
			dropout_rate: float = 0,
			hidden_activation: typing.Optional[nn.Module] = None,
			init_fn: typing.Optional[typing.Callable] = None,
			norm: typing.Union[bool, typing.List[bool]] = False
	):
		super(LinearModel, self).__init__()
		# Save the arguments
		self.args = {
			'block_size': block_size,
			'vocab_size': vocab_size,
			'layer_sizes': layer_sizes,
			'dropout_rate': dropout_rate,
			'hidden_activation': hidden_activation.__class__.__name__ if hidden_activation else None,
			'init_fn': init_fn.__name__ if init_fn else None,
			'norm': norm
		}
		self.input_size = block_size
		self.output_size = vocab_size

		layer_sizes = [block_size] + layer_sizes + [vocab_size]

		if isinstance(norm, bool):
			norm = [norm for _ in range(len(layer_sizes) - 1)]
		if len(norm) != len(layer_sizes) - 1:
			raise ValueError("Norm size doesn't match layers size")
		self.layers = nn.ModuleList()

		for i in range(len(layer_sizes) - 1):
			if norm[i]:
				self.layers.append(nn.LayerNorm(layer_sizes[i]))
			self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if init_fn is not None:
				init_fn(self.layers[-1].weight)
		if hidden_activation is None:
			hidden_activation = nn.Identity()
		self.hidden_activation = hidden_activation

		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = layer.forward(x)
			x = self.hidden_activation(x)
			x = self.dropout(x)
		x = self.layers[-1](x)
		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args

	@classmethod
	def import_config(cls, config: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
		if config['hidden_activation']:
			hidden_activation_module = importlib.import_module('torch.nn')  # replace with the actual module
			config['hidden_activation'] = getattr(hidden_activation_module, config['hidden_activation'])()
		if config['init_fn']:
			init_fn_module = importlib.import_module('torch.nn.init')  # replace with the actual module
			config['init_fn'] = getattr(init_fn_module, config['init_fn'])
		return config
