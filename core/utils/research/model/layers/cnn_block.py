import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import Indicators, DynamicLayerNorm
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CNNBlock(SpinozaModule):

	def __init__(
			self,
			input_channels: int,
			conv_channels: typing.List[int],
			kernel_sizes: typing.List[int],
			indicators: typing.Optional[Indicators] = None,
			pool_sizes: typing.Optional[typing.List[int]] = None,
			hidden_activation: typing.Optional[nn.Module] = None,
			dropout_rate: typing.Union[float, typing.List[float]] = 0,
			init_fn: typing.Optional[nn.Module] = None,
			padding: int = 1,
			norm: typing.Union[bool, typing.List[bool]] = False,
			stride: typing.Union[int, typing.List[int]] = 1,
	):
		super(CNNBlock, self).__init__()
		self.args = {
			'conv_channels': conv_channels,
			'kernel_sizes': kernel_sizes,
			'pool_sizes': pool_sizes,
			'hidden_activation': hidden_activation.__class__.__name__ if hidden_activation else None,
			'init_fn': init_fn.__name__ if init_fn else None,
			'dropout_rate': dropout_rate,
			'padding': padding,
			'norm': norm,
			'stride': stride,
			'indicators': indicators
		}
		self.layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()

		if indicators is None:
			indicators = Indicators()
		self.indicators = indicators

		if pool_sizes is None:
			pool_sizes = [
				0
				for _ in kernel_sizes
			]
		conv_channels = [input_channels] + conv_channels  # TODO: FIX INDICATORS AND APPLY: [self.indicators.indicators_len] + conv_channels

		if isinstance(norm, bool):
			norm = [norm for _ in range(len(conv_channels) - 1)]
		if len(norm) != len(conv_channels) - 1:
			raise ValueError("Norm size doesn't match layers size")

		if isinstance(dropout_rate, (int, float)):
			dropout_rate = [dropout_rate for _ in range(len(conv_channels)-1)]
		if len(dropout_rate) != (len(conv_channels)-1):
			raise ValueError("Dropout size doesn't match layers size")

		if isinstance(stride, int):
			stride = [stride for _ in kernel_sizes]
		if len(stride) != len(kernel_sizes):
			raise ValueError("Stride size doesn't match layers size")

		self.layers = nn.ModuleList(self._build_conv_layers(
			channels=conv_channels,
			kernel_sizes=kernel_sizes,
			stride=stride,
			padding=padding
		))

		for i in range(len(conv_channels) - 1):
			if norm[i]:
				self.norm_layers.append(DynamicLayerNorm())
			else:
				self.norm_layers.append(nn.Identity())

			if init_fn is not None:
				init_fn(self.layers[i].weight)

			if pool_sizes[i] > 0:
				self.pool_layers.append(nn.AvgPool1d(kernel_size=pool_sizes[i], stride=2))
			else:
				self.pool_layers.append(nn.Identity())
		self.hidden_activation = hidden_activation

		self.dropouts = nn.ModuleList([
			nn.Dropout(rate) if rate > 0 else nn.Identity()
			for rate in dropout_rate
		])

	@staticmethod
	def _build_conv_layers(
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

	def call(self, x):

		out = x  # TODO: FIX INDICATORS DIM ISSUE AND out = self.indicators(x)

		for layer, pool_layer, norm, dropout in zip(self.layers, self.pool_layers, self.norm_layers, self.dropouts):
			out = norm(out)
			out = layer.forward(out)
			out = self.hidden_activation(out)
			out = pool_layer(out)
			out = dropout(out)

		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
