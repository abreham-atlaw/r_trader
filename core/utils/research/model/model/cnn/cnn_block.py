import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import DynamicLayerNorm, DynamicPool
from core.utils.research.model.model.savable import SpinozaModule


class CNNBlock(SpinozaModule):

	def __init__(
			self,
			input_channels: int,
			conv_channels: typing.List[int],
			kernel_sizes: typing.List[int],
			pool_sizes: typing.List[typing.Union[int, typing.Tuple[int, int, int]]] = None,
			hidden_activation: typing.Optional[nn.Module] = None,
			dropout_rate: typing.Union[float, typing.List[float]] = 0,
			norm: typing.Union[bool, typing.List[bool]] = False,
			stride: typing.Union[int, typing.List[int]] = 1,
			padding: int = 0,
			avg_pool=True,
			init_fn: typing.Optional[nn.Module] = None,
	):

		self.args = {
			'input_channels': input_channels,
			'conv_channels': conv_channels,
			'kernel_sizes': kernel_sizes,
			'pool_sizes': pool_sizes,
			'hidden_activation': hidden_activation,
			'dropout_rate': dropout_rate,
			'norm': norm,
			'stride': stride,
			'padding': padding,
			'avg_pool': avg_pool,
			'init_fn': init_fn.__name__ if init_fn else None,
		}

		super(CNNBlock, self).__init__(auto_build=False)

		conv_channels = [input_channels] + conv_channels

		pool_sizes = self.__prepare_arg_pool(pool_sizes)
		norm = self.__prepare_arg_norm(norm, kernel_sizes)
		dropout_rate = self.__prepare_arg_dropout(dropout_rate, kernel_sizes)
		stride = self.__prepare_arg_stride(stride, kernel_sizes)
		hidden_activation = self.__prepare_arg_hidden_activation(hidden_activation, kernel_sizes)

		self.layers = nn.ModuleList(self._build_conv_layers(
			channels=conv_channels,
			kernel_sizes=kernel_sizes,
			stride=stride,
			padding=padding
		))
		self.pool_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()

		for i in range(len(conv_channels) - 1):
			if norm[i]:
				self.norm_layers.append(DynamicLayerNorm())
			else:
				self.norm_layers.append(nn.Identity())
			if init_fn is not None:
				init_fn(self.layers[-1].weight)

			self.pool_layers.append(
				DynamicPool(
					pool_range=(pool_sizes[i][0], pool_sizes[i][1]),
					pool_size=pool_sizes[i][2]
				) if pool_sizes[i][-1] > 0
				else nn.Identity()
			)

		self.hidden_activations = nn.ModuleList(hidden_activation)

		self.dropouts = nn.ModuleList([
			nn.Dropout(rate) if rate > 0 else nn.Identity()
			for rate in dropout_rate
		])

	def __prepare_arg_pool(self, pool_sizes: typing.List[typing.Union[int, typing.Tuple[int, int, int]]]) -> typing.List[typing.Tuple[int, int, int]]:
		if pool_sizes is None:
			pool_sizes = [
				0
				for _ in self.kernel_sizes
			]
		pool_sizes = [
			ps
			if isinstance(ps, typing.Iterable)
			else (0, 1, ps)
			for ps in pool_sizes
		]
		return pool_sizes

	@staticmethod
	def __prepare_arg_hidden_activation(
			hidden_activation: typing.Union[typing.List[nn.Module], nn.Module],
			kernel_sizes: typing.List[int]
	) -> typing.List[nn.Module]:
		if isinstance(hidden_activation, nn.Module):
			hidden_activation = [hidden_activation for _ in kernel_sizes]
		if len(hidden_activation) != len(kernel_sizes):
			raise ValueError("Hidden activation size doesn't match layers size")
		return hidden_activation

	@staticmethod
	def __prepare_arg_stride(
			stride: typing.Union[int, typing.List[int]],
			kernel_sizes: typing.List[int]
	):
		if isinstance(stride, int):
			stride = [stride for _ in kernel_sizes]
		if len(stride) != len(kernel_sizes):
			raise ValueError("Stride size doesn't match layers size")
		return stride

	@staticmethod
	def __prepare_arg_dropout(
			dropout_rate: typing.Union[int, typing.List[int]],
			kernel_sizes: typing.List[int]
	) -> typing.List[float]:
		if isinstance(dropout_rate, (int, float)):
			dropout_rate = [dropout_rate for _ in range(len(kernel_sizes))]
		if len(dropout_rate) != len(kernel_sizes):
			raise ValueError("Dropout size doesn't match layers size")
		return dropout_rate

	@staticmethod
	def __prepare_arg_norm(
			norm: typing.Union[bool, typing.List[bool]],
			kernel_sizes: typing.List[int]
	) -> typing.List[bool]:
		if isinstance(norm, bool):
			norm = [norm for _ in range(len(kernel_sizes))]
		if len(norm) != len(kernel_sizes):
			raise ValueError("Norm size doesn't match layers size")
		return norm

	def _build_conv_layers(
			self,
			channels: typing.List[int],
			kernel_sizes: typing.List[int],
			stride: typing.List[int],
			padding: int
	) -> typing.List[nn.Module]:
		return [
			nn.Conv1d(
				in_channels=channels[i],
				out_channels=channels[i + 1],
				kernel_size=kernel_sizes[i],
				stride=stride[i],
				padding=padding,
			)
			for i in range(len(channels) - 1)
		]

	def call(self, x: torch.Tensor) -> torch.Tensor:
		out = x
		for (
				layer, pool_layer, norm, dropout, hidden_activation
		) in zip(
			self.layers, self.pool_layers, self.norm_layers, self.dropouts, self.hidden_activations
		):
			out = norm(out)
			out = layer(out)
			out = hidden_activation(out)
			out = pool_layer(out)
			out = dropout(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
