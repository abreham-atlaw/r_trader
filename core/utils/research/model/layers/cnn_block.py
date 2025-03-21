import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CNNBlock(SpinozaModule):

	def __init__(
			self,
			conv_channels: typing.List[int],
			kernel_sizes: typing.List[int],
			indicators: typing.Optional[Indicators] = None,
			pool_sizes: typing.Optional[typing.List[int]] = None,
			hidden_activation: typing.Optional[nn.Module] = None,
			dropout_rate: float = 0,
			init_fn: typing.Optional[nn.Module] = None,
			padding: int = 1,
			avg_pool=True,
			input_size: int = 1028,
			norm: typing.Union[bool, typing.List[bool]] = False,
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
			'input_size': input_size,
			'norm': norm,
			'indicators': indicators
		}
		self.layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()
		self.input_size = input_size

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

		for i in range(len(conv_channels) - 1):
			if norm[i]:
				self.norm_layers.append(nn.BatchNorm1d(conv_channels[i]))
			else:
				self.norm_layers.append(nn.Identity())
			self.layers.append(
				nn.Conv1d(
					in_channels=conv_channels[i],
					out_channels=conv_channels[i + 1],
					kernel_size=kernel_sizes[i],
					stride=1,
					padding=padding
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

		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def call(self, x):
		out = self.indicators(x)
		for layer, pool_layer, norm in zip(self.layers, self.pool_layers, self.norm_layers):
			out = norm(out)
			out = layer.forward(out)
			out = self.hidden_activation(out)
			out = pool_layer(out)
			out = self.dropout(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
