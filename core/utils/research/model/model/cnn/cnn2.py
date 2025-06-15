import typing

from torch import nn

from core.utils.research.model.model.cnn.bridge_block import BridgeBlock
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.savable import SpinozaModule


class CNN2(SpinozaModule):

	def __init__(
			self,
			extra_len: int,
			embedding_block: EmbeddingBlock,
			cnn_block: CNNBlock,
			collapse_block: CollapseBlock,
			input_size: int = 1028,
			bridge_block: typing.Optional[BridgeBlock] = None
	):
		super(CNN2, self).__init__(input_size=input_size, auto_build=False)
		self.args = {
			'extra_len': extra_len,
			'embedding_block': embedding_block,
			'cnn_block': cnn_block,
			'collapse_block': collapse_block,
			'input_size': input_size,
			'bridge_block': bridge_block
		}
		self.extra_len = extra_len
		self.embedding_block = embedding_block
		self.cnn_block = cnn_block
		self.collapse_block = collapse_block
		self.bridge_block = bridge_block if bridge_block is not None else nn.Identity()
		self.init()

	def call(self, x):
		seq = x[:, :-self.extra_len]

		embedded = self.embedding_block(seq)

		cnn_out = self.cnn_block(embedded)

		bridge_out = self.bridge_block(cnn_out)

		out = self.collapse_block(bridge_out, extra=x[:, -self.extra_len:])

		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
