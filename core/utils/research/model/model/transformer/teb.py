import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.savable import SpinozaModule


class TransformerEmbeddingBlock(SpinozaModule):

	def __init__(
			self,
			embedding_block: EmbeddingBlock = None,
			cnn_block: CNNBlock = None
	):
		self.args = {
			'embedding_block': embedding_block,
			'cnn_block': cnn_block
		}
		super().__init__()

		self.embedding = embedding_block if embedding_block is not None else nn.Identity()
		self.cnn = cnn_block if cnn_block is not None else nn.Identity()

	def call(self, *args, **kwargs) -> torch.Tensor:
		embedded = self.embedding(*args, **kwargs)
		out = self.cnn(embedded)
		out = out.transpose(1, 2)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
