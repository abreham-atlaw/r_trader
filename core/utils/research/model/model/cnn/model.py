import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SavableModel


class CNN(SavableModel):

	def __init__(
			self,
			num_classes: int,
			extra_len: int,
			conv_channels: typing.List[int],
			kernel_sizes: typing.List[int],
			ff_linear: LinearModel = None,
			pool_sizes: typing.Optional[typing.List[int]] = None,
			hidden_activation: typing.Optional[nn.Module] = None,
			dropout_rate: float = 0,
			init_fn: typing.Optional[nn.Module] = None,
			padding: int = 1,
			avg_pool=True,
			linear_collapse=False,
			input_size: int = 1028
	):
		super(CNN, self).__init__()
		self.args = {
			'extra_len': extra_len,
			'ff_linear': ff_linear,
			'num_classes': num_classes,
			'conv_channels': conv_channels,
			'kernel_sizes': kernel_sizes,
			'pool_sizes': pool_sizes,
			'hidden_activation': hidden_activation.__class__.__name__ if hidden_activation else None,
			'init_fn': init_fn.__name__ if init_fn else None,
			'dropout_rate': dropout_rate,
			'padding': padding,
			'avg_pool': avg_pool,
			'linear_collapse': linear_collapse,
			'input_size': input_size
		}
		self.extra_len = extra_len
		self.layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		self.input_size = input_size

		if pool_sizes is None:
			pool_sizes = [
				0
				for _ in kernel_sizes
			]
		conv_channels = [1] + conv_channels

		for i in range(len(conv_channels) - 1):
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
		if ff_linear is None:
			self.fc = nn.Linear(conv_channels[-1]+self.extra_len, num_classes)
		else:
			self.fc = nn.Sequential(
				nn.Linear(conv_channels[-1] + self.extra_len, ff_linear.input_size),
				ff_linear,
				nn.Linear(ff_linear.output_size, num_classes)
			)

		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

		self.collapse_layer = None if linear_collapse else nn.AdaptiveAvgPool1d((1,))

	def collapse(self, out: torch.Tensor) -> torch.Tensor:
		if self.collapse_layer is None:
			self.collapse_layer = nn.Linear(out.shape[-1], 1)
		return self.collapse_layer(out)

	def forward(self, x):
		seq = x[:, :-self.extra_len]
		out = torch.unsqueeze(seq, dim=1)
		for layer, pool_layer in zip(self.layers, self.pool_layers):
			out = layer.forward(out)
			out = self.hidden_activation(out)
			out = pool_layer(out)
			out = self.dropout(out)
		out = self.collapse(out)
		out = out.reshape(out.size(0), -1)
		out = self.dropout(out)
		out = torch.cat((out, x[:, -self.extra_len:]), dim=1)
		out = self.fc(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
