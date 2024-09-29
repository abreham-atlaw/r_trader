import importlib
import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class LinearModel(SpinozaModule):

	def __init__(
			self,
			layer_sizes: typing.List[int],
			dropout_rate: float = 0,
			hidden_activation: typing.Optional[nn.Module] = None,
			init_fn: typing.Optional[typing.Callable] = None,
			norm: typing.Union[bool, typing.List[bool]] = False,
			input_size: int = None,
			softmax: bool = True
	):
		super(LinearModel, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'layer_sizes': layer_sizes,
			'dropout_rate': dropout_rate,
			'hidden_activation': hidden_activation.__class__.__name__ if hidden_activation else None,
			'init_fn': init_fn.__name__ if init_fn else None,
			'norm': norm,
			'input_size': input_size,
			'softmax': softmax
		}
		self.output_size = layer_sizes[-1]
		self.layers_sizes = [input_size] + layer_sizes
		self.init_fn = init_fn

		self.softmax = nn.Softmax(dim=1) if softmax else nn.Identity()

		self.layers = None
		self.norms = None

		self.norms_mask = norm
		if isinstance(norm, bool):
			self.norms_mask = [norm for _ in range(len(layer_sizes) - 1)]
		if len(norm) != len(self.layers_sizes) - 1:
			raise ValueError("Norm size doesn't match layers size")

		if hidden_activation is None:
			hidden_activation = nn.Identity()
		self.hidden_activation = hidden_activation

		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()
		if input_size is not None:
			self.init()

	def build(self, input_size: torch.Size):
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()

		if self.layers_sizes[0] is None:
			self.layers_sizes[0] = input_size[-1]

		for i in range(len(self.layers_sizes) - 1):

			if self.norms_mask[i]:
				self.norms.append(nn.LayerNorm(self.layers_sizes[i]))
			else:
				self.norms.append(nn.Identity())

			self.layers.append(
				nn.Linear(
					self.layers_sizes[i],
					self.layers_sizes[i + 1]
				)
			)
			if self.init_fn is not None:
				self.init_fn(self.layers[-1].weight)

	def call(self, x):
		out = x
		for norm, layer, i in zip(self.norms, self.layers, range(len(self.layers))):
			out = norm(out)
			out = layer.forward(out)
			if i == len(self.layers) - 1:
				continue
			out = self.hidden_activation(out)
			out = self.dropout(out)
		out = self.softmax(out)
		return out

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
