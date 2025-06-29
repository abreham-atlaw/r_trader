import importlib
import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import DynamicLayerNorm, PassThroughLayer
from core.utils.research.model.model.savable import SpinozaModule


class LinearModel(SpinozaModule):

	def __init__(
			self,
			layer_sizes: typing.List[int],
			dropout_rate: typing.Union[float, typing.List[float]] = 0,
			hidden_activation: typing.Union[nn.Module, typing.List[nn.Module]] = None,
			init_fn: typing.Optional[typing.Callable] = None,
			norm: typing.Union[typing.Union[bool, typing.List[bool]], typing.Union[nn.Module, typing.List[nn.Module]]] = False,
			input_size: int = None,
			**kwargs
	):
		super(LinearModel, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'layer_sizes': layer_sizes,
			'dropout_rate': dropout_rate.copy() if isinstance(dropout_rate, list) else dropout_rate,
			'hidden_activation': hidden_activation.copy() if isinstance(hidden_activation, list) else hidden_activation,
			'init_fn': init_fn.__name__ if init_fn else None,
			'norm': norm,
			'input_size': input_size
		}
		self.output_size = layer_sizes[-1]
		self.layers_sizes = [input_size] + layer_sizes
		self.init_fn = init_fn

		self.layers = None

		num_layers = len(self.layers_sizes) - 1

		if isinstance(dropout_rate, (int, float)):
			dropout_rate = [dropout_rate for _ in range(len(self.layers_sizes) - 2)]
		dropout_rate += [0]
		if len(dropout_rate) != (len(self.layers_sizes) - 1):
			raise ValueError("Dropout size doesn't match layers size")

		self.dropouts = nn.ModuleList([
			nn.Dropout(rate) if rate > 0 else nn.Identity()
			for rate in dropout_rate
		])

		self.norms = self.__prepare_arg_norm(norm, num_layers)
		self.hidden_activations = self.__prepare_arg_hidden_activation(hidden_activation, num_layers)

		if input_size is not None:
			self.init()

	@staticmethod
	def __prepare_arg_norm(norm, num_layers) -> typing.List[nn.Module]:
		if norm is None:
			norm = False
		if isinstance(norm, bool):
			norm = DynamicLayerNorm() if norm else nn.Identity()
		if isinstance(norm, nn.Module) or not isinstance(norm, typing.Iterable):
			norm = [norm for _ in range(num_layers)]
		if (not isinstance(norm, nn.Module)) and isinstance(norm, typing.Iterable) and len(norm) > 0 and isinstance(norm[0], bool):
			norm = [DynamicLayerNorm() if n else nn.Identity() for n in norm]
		if len(norm) != num_layers:
			raise ValueError("Norm size doesn't match layers size")
		return nn.ModuleList(norm)

	@staticmethod
	def __prepare_arg_dropout(dropout_rate, num_layers) -> typing.List[nn.Module]:
		if isinstance(dropout_rate, (int, float)):
			dropout_rate = [dropout_rate for _ in range(len(num_layers) - 1)]
		if len(dropout_rate) == num_layers-1:
			dropout_rate += [0]
		if len(dropout_rate) != num_layers:
			raise ValueError("Dropout size doesn't match layers size")

		dropouts = nn.ModuleList([
			nn.Dropout(rate) if rate > 0 else nn.Identity()
			for rate in dropout_rate
		])

		return dropouts

	@staticmethod
	def __prepare_arg_hidden_activation(hidden_activation, num_layers) -> typing.List[nn.Module]:
		if hidden_activation is None:
			hidden_activation = nn.Identity()
		if isinstance(hidden_activation, nn.Module):
			hidden_activation = [hidden_activation for _ in range(num_layers - 1)]
		if len(hidden_activation) == num_layers - 1:
			hidden_activation += [nn.Identity()]
		if len(hidden_activation) != num_layers:
			raise ValueError("Hidden activation size doesn't match layers size")
		hidden_activation = nn.ModuleList(hidden_activation)
		return hidden_activation

	def build(self, input_size: torch.Size):
		self.layers = nn.ModuleList()

		if self.layers_sizes[0] is None:
			self.layers_sizes[0] = input_size[-1]

		for i in range(len(self.layers_sizes) - 1):

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
		for (
				norm, layer, dropout, hidden_activation, i
		) in zip(
			self.norms, self.layers, self.dropouts, self.hidden_activations, range(len(self.layers))
		):
			out = norm(out)
			out = layer(out)
			out = hidden_activation(out)
			out = dropout(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
