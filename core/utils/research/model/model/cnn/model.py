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
			input_shape: typing.Tuple[int, int],
			extra_len: int,
			cnn_block: CNNBlock,
			collapse_block: CollapseFFBlock,
			channel_ff_block: LinearModel = None,
	):
		super(CNN, self).__init__(input_size=input_shape, auto_build=False)
		self.args = {
			'extra_len': extra_len,
			'input_shape': input_shape,
			'cnn_block': cnn_block,
			'channel_ff_block': channel_ff_block,
			'collapse_block': collapse_block
		}
		self.extra_len = extra_len
		self.cnn_block = cnn_block
		self.channel_ff_block = AxisFFN(channel_ff_block, axis=1) if channel_ff_block is not None else nn.Identity()
		self.collapse_block = collapse_block
		self.init()

	def call(self, x):
		seq = x[:, :, :-self.extra_len]
		extra = x[:, :, -self.extra_len:]

		cnn_out = self.cnn_block(seq)
		channel_ff_out = self.channel_ff_block(cnn_out)

		out = self.collapse_block(channel_ff_out, extra)

		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
