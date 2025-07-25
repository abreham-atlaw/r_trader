import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

from core.utils.research.model.layers import Indicators, DynamicLayerNorm, Axis, DynamicPool, \
	FlattenLayer
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


# THIS IS LEGACY CODE. USE cnn.py instead.

class CNN(SpinozaModule):

	def __init__(
			self,
			extra_len: int,
			conv_channels: typing.List[int],
			kernel_sizes: typing.List[int],
			ff_block: LinearModel = None,
			indicators: typing.Optional[Indicators] = None,
			pool_sizes: typing.Optional[typing.List[typing.Union[int, typing.Tuple[int, int, int]]]] = None,
			hidden_activation: typing.Union[nn.Module, typing.List[nn.Module]] = None,
			dropout_rate: typing.Union[float, typing.List[float]] = 0,
			init_fn: typing.Optional[nn.Module] = None,
			padding: int = 1,
			avg_pool=True,
			linear_collapse=False,
			input_size: int = 1028,
			norm: typing.Union[bool, typing.List[bool]] = False,
			stride: typing.Union[int, typing.List[int]] = 1,
			positional_encoding: bool = False,
			norm_positional_encoding: bool = False,
			channel_ffn: typing.Optional[nn.Module] = None,
			input_dropout: float = 0.0,
			collapse_avg_pool: bool = False,
			input_norm: bool = False
	):
		super(CNN, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'extra_len': extra_len,
			'ff_block': ff_block,
			'conv_channels': conv_channels,
			'kernel_sizes': kernel_sizes,
			'pool_sizes': pool_sizes,
			'stride': stride,
			'hidden_activation': hidden_activation,
			'init_fn': init_fn.__name__ if init_fn else None,
			'dropout_rate': dropout_rate,
			'padding': padding,
			'avg_pool': avg_pool,
			'linear_collapse': linear_collapse,
			'input_size': input_size,
			'norm': norm,
			'indicators': indicators,
			'positional_encoding': positional_encoding,
			'norm_positional_encoding': norm_positional_encoding,
			'channel_ffn': channel_ffn,
			'input_dropout': input_dropout,
			'input_norm': input_norm,
			"collapse_avg_pool": collapse_avg_pool
		}
		self.extra_len = extra_len
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
		pool_sizes = [
			ps
			if isinstance(ps, typing.Iterable)
			else (0, 1, ps)
			for ps in pool_sizes
		]
		conv_channels = [self.indicators.indicators_len] + conv_channels

		if hidden_activation is None:
			hidden_activation = nn.Identity()


		if isinstance(norm, bool):
			norm = [norm for _ in range(len(conv_channels) - 1)]
		if len(norm) != len(conv_channels) - 1:
			raise ValueError("Norm size doesn't match layers size")

		if isinstance(dropout_rate, (int, float)):
			dropout_rate = [dropout_rate for _ in range(len(conv_channels))]
		if len(dropout_rate) != len(conv_channels):
			raise ValueError("Dropout size doesn't match layers size")

		if isinstance(stride, int):
			stride = [stride for _ in kernel_sizes]
		if len(stride) != len(kernel_sizes):
			raise ValueError("Stride size doesn't match layers size")

		if isinstance(hidden_activation, nn.Module):
			hidden_activation = [hidden_activation for _ in conv_channels[:-1]]
		if len(hidden_activation) != len(conv_channels) - 1:
			raise ValueError("Hidden activation size doesn't match layers size")

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
				init_fn(self.layers[-1].weight)
			if pool_sizes[i][-1] > 0:
				pool = DynamicPool(
					pool_range=(pool_sizes[i][0], pool_sizes[i][1]),
					pool_size=pool_sizes[i][2]
				)
				self.pool_layers.append(pool)
			else:
				self.pool_layers.append(nn.Identity())

		self.hidden_activations = nn.ModuleList(hidden_activation)

		self.dropouts = nn.ModuleList([
			nn.Dropout(rate) if rate > 0 else nn.Identity()
			for rate in dropout_rate
		])

		self.ff_block = ff_block
		self.collapse_layer = None if linear_collapse else nn.AdaptiveAvgPool1d((1,))

		self.pos_layer = None

		self.pos_norm = DynamicLayerNorm() if norm_positional_encoding else nn.Identity()
		self.pos = self.positional_encoding if positional_encoding else nn.Identity()

		if positional_encoding:
			self.pos = self.positional_encoding

		else:
			self.pos = nn.Identity()

		self.channel_ffn = Axis(channel_ffn, axis=1) if channel_ffn else nn.Identity()
		self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
		self.input_norm = DynamicLayerNorm() if input_norm else nn.Identity()
		self.collapse_avg_pool = nn.AdaptiveAvgPool1d(1) if collapse_avg_pool else nn.Identity()
		self.flatten = FlattenLayer(1, 2)
		self.init()

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

	def positional_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
		if self.pos_layer is None:
			self.pos_layer = PositionalEncodingPermute1D(inputs.shape[1])
		inputs = self.pos_norm(inputs)
		return inputs + self.pos_layer(inputs)

	def collapse(self, out: torch.Tensor) -> torch.Tensor:
		out = self.collapse_avg_pool(out)
		return self.flatten(out)

	def call(self, x):
		seq = x[:, :-self.extra_len]

		out = self.input_norm(seq)

		out = self.indicators(out)

		out = self.pos(out)

		out = self.input_dropout(out)

		for (
				layer, pool_layer, norm, hidden_activation, dropout
		) in zip(
			self.layers, self.pool_layers, self.norm_layers, self.hidden_activations, self.dropouts
		):
			out = norm(out)
			out = layer(out)
			out = hidden_activation(out)
			out = pool_layer(out)
			out = dropout(out)
		out = self.channel_ffn(out)
		out = self.collapse(out)
		out = out.reshape(out.size(0), -1)
		out = self.dropouts[-1](out)
		out = torch.cat((out, x[:, -self.extra_len:]), dim=1)
		out = self.ff_block(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args