import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

from core.utils.research.model.layers import Indicators, DynamicLayerNorm
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CNN2(SpinozaModule):

	def __init__(
			self,
			extra_len: int,
			embedding_block: EmbeddingBlock,
			cnn_block: CNNBlock,
			collapse_block: CollapseBlock,
			input_size: int = 1028,
	):
		super(CNN2, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'extra_len': extra_len,
			'embedding_block': embedding_block,
			'cnn_block': cnn_block,
			'collapse_block': collapse_block,
			'input_size': input_size,
		}
		self.extra_len = extra_len
		self.embedding_block = embedding_block
		self.cnn_block = cnn_block
		self.collapse_block = collapse_block
		self.init()

	def call(self, x):
		seq = x[:, :-self.extra_len]

		embedded = self.embedding_block(seq)

		cnn_out = self.cnn_block(embedded)

		out = self.collapse_block(cnn_out, extra=x[:, -self.extra_len:])

		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
