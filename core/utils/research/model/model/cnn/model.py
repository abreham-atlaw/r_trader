import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

from core.utils.research.model.layers import Indicators, DynamicLayerNorm
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CNN(SpinozaModule):

	def __init__(
			self,
			extra_len: int,
			conv_channels: typing.List[int],
			kernel_sizes: typing.List[int],
			ff_block: LinearModel = None,
			indicators: typing.Optional[Indicators] = None,
			pool_sizes: typing.Optional[typing.List[int]] = None,
			hidden_activation: typing.Optional[nn.Module] = None,
			dropout_rate: typing.Union[float, typing.List[float]] = 0,
			init_fn: typing.Optional[nn.Module] = None,
			padding: int = 1,
			avg_pool=True,
			linear_collapse=False,
			input_size: int = 1028,
			norm: typing.Union[bool, typing.List[bool]] = False,
			stride: typing.Union[int, typing.List[int]] = 1,
			positional_encoding: bool = False,
			norm_positional_encoding: bool = False
	):
		super(CNN, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'extra_len': extra_len,
			'ff_block': ff_block,
			'conv_channels': conv_channels,
			'kernel_sizes': kernel_sizes,
			'pool_sizes': pool_sizes,
			'hidden_activation': hidden_activation.__class__.__name__ if hidden_activation else None,
			'init_fn': init_fn.__name__ if init_fn else None,
			'dropout_rate': dropout_rate,
			'padding': padding,
			'avg_pool': avg_pool,
			'linear_collapse': linear_collapse,
			'input_size': input_size,
			'norm': norm,
			'indicators': indicators,
			'positional_encoding': positional_encoding,
			'norm_positional_encoding': norm_positional_encoding
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
		conv_channels = [self.indicators.indicators_len] + conv_channels

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

		for i in range(len(conv_channels) - 1):
			if norm[i]:
				self.norm_layers.append(DynamicLayerNorm())
			else:
				self.norm_layers.append(nn.Identity())
			self.layers.append(
				nn.Conv1d(
					in_channels=conv_channels[i],
					out_channels=conv_channels[i + 1],
					kernel_size=kernel_sizes[i],
					stride=stride[i],
					padding=padding,
				)
			)
			if init_fn is not None:
				init_fn(self.layers[-1].weight)
			if pool_sizes[i] > 0:
				if avg_pool:
					pool = nn.AvgPool1d(kernel_size=pool_sizes[i], stride=2)
				else:
					pool = nn.MaxPool1d(kernel_size=pool_sizes[i], stride=2)
				self.pool_layers.append(pool)
			else:
				self.pool_layers.append(nn.Identity())
		self.hidden_activation = hidden_activation

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

		self.init()

	def positional_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
		if self.pos_layer is None:
			self.pos_layer = PositionalEncodingPermute1D(inputs.shape[1])
		inputs = self.pos_norm(inputs)
		return inputs + self.pos_layer(inputs)

	def collapse(self, out: torch.Tensor) -> torch.Tensor:
		return torch.flatten(out, 1, 2)

	def call(self, x):
		seq = x[:, :-self.extra_len]
		out = self.indicators(seq)

		out = self.pos(out)

		for layer, pool_layer, norm, dropout in zip(self.layers, self.pool_layers, self.norm_layers, self.dropouts):
			out = norm(out)
			out = layer.forward(out)
			out = self.hidden_activation(out)
			out = pool_layer(out)
			out = dropout(out)
		out = self.collapse(out)
		out = out.reshape(out.size(0), -1)
		out = self.dropouts[-1](out)
		out = torch.cat((out, x[:, -self.extra_len:]), dim=1)
		out = self.ff_block(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
