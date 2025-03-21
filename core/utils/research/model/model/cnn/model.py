import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

from core.utils.research.model.layers import Indicators
from core.utils.research.model.layers.cnn_block import CNNBlock
from core.utils.research.model.layers.collapse_ff_block import CollapseFFBlock
from core.utils.research.model.layers import Indicators, DynamicLayerNorm, AxisFFN
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
			norm_positional_encoding: bool = False,
			channel_ffn: typing.Optional[nn.Module] = None
	):
		super(CNN, self).__init__()
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
			'indicators': indicators
		}
		self.extra_len = extra_len
		self.input_size = input_size

		self.cnn_block = CNNBlock(
			conv_channels,
			kernel_sizes,
			indicators,
			pool_sizes,
			hidden_activation,
			dropout_rate,
			init_fn,
			padding,
			avg_pool,
			input_size,
			norm,
		)

		self.collapse_block = CollapseFFBlock(
			num_classes,
			conv_channels[-1],
			extra_len,
			ff_linear,
			linear_collapse,
		)
		self.__init()

	def __init(self):
		init_data = torch.rand((1, self.input_size))
		self(init_data)

	def forward(self, x):
		seq = x[:, :-self.extra_len]
		out = self.cnn_block(seq)
		out = self.collapse_block(out, x[:, -self.extra_len:])
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
