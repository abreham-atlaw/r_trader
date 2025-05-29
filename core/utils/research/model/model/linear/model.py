import importlib
import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import PassThroughLayer
from core.utils.research.model.model.savable import SpinozaModule


class LinearModel(SpinozaModule):

	def __init__(
			self,
			layer_sizes: typing.List[int],
			dropout_rate: typing.Union[float, typing.List[float]] = 0,
			hidden_activation: typing.Union[nn.Module, typing.List[nn.Module]] = None,
			init_fn: typing.Optional[typing.Callable] = None,
			norm: typing.Union[bool, typing.List[bool]] = False,
			input_size: int = None,
			norm_learnable: bool = True
	):
		super(LinearModel, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'layer_sizes': layer_sizes,
			'dropout_rate': dropout_rate.copy() if isinstance(dropout_rate, list) else dropout_rate,
			'hidden_activation': hidden_activation.copy() if isinstance(hidden_activation, list) else hidden_activation,
			'init_fn': init_fn.__name__ if init_fn else None,
			'norm': norm,
			'norm_learnable': norm_learnable,
			'input_size': input_size
		}
		self.output_size = layer_sizes[-1]
		self.layers_sizes = [input_size] + layer_sizes
		self.init_fn = init_fn

		self.layers = None
		self.norms = None

		num_layers = len(self.layers_sizes) - 1

		self.norms_mask = norm
		if isinstance(norm, bool):
			self.norms_mask = [norm for _ in range(len(self.layers_sizes) - 1)]
		if len(self.norms_mask) != (len(self.layers_sizes) - 1):
			raise ValueError("Norm size doesn't match layers size")
		self.norm_learnable = norm_learnable

		if isinstance(dropout_rate, (int, float)):
			dropout_rate = [dropout_rate for _ in range(len(self.layers_sizes) - 2)]
		dropout_rate += [0]
		if len(dropout_rate) != (len(self.layers_sizes) - 1):
			raise ValueError("Dropout size doesn't match layers size")

		self.dropouts = nn.ModuleList([
			nn.Dropout(rate) if rate > 0 else nn.Identity()
			for rate in dropout_rate
		])

		self.hidden_activations = self.__prepare_arg_hidden_activation(hidden_activation, num_layers)

		if input_size is not None:
			self.init()

	@staticmethod
	def __prepare_arg_hidden_activation(hidden_activation, num_layers) -> typing.List[nn.Module]:
		if hidden_activation is None:
			hidden_activation = nn.Identity()
		if isinstance(hidden_activation, nn.Module):
			hidden_activation = [hidden_activation for _ in range(num_layers - 1)]
		hidden_activation += [nn.Identity()]
		if len(hidden_activation) != num_layers:
			raise ValueError("Hidden activation size doesn't match layers size")
		hidden_activation = nn.ModuleList([ha if isinstance(ha, nn.Identity) else PassThroughLayer(ha) for ha in hidden_activation])
		return hidden_activation

	def build(self, input_size: torch.Size):
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()

		if self.layers_sizes[0] is None:
			self.layers_sizes[0] = input_size[-1]

		for i in range(len(self.layers_sizes) - 1):

			if self.norms_mask[i]:
				self.norms.append(
					nn.LayerNorm(
						self.layers_sizes[i],
						elementwise_affine=self.norm_learnable
					)
				)
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
		for (
				norm, layer, dropout, hidden_activation, i
		) in zip(
			self.norms, self.layers, self.dropouts, self.hidden_activations, range(len(self.layers))
		):
			out = norm(out)
			out = layer(out)
			if i == len(self.layers) - 1:
				continue
			out = hidden_activation(out)
			out = dropout(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
