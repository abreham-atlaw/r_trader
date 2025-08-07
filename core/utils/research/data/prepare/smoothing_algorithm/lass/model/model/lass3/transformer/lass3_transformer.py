import typing

import torch

from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import TransformerEmbeddingBlock, DecoderBlock
from .lass3_decoder_block import Lass3DecoderBlock


class Lass3Transformer(SpinozaModule):

	def __init__(
			self,
			block_size: int,
			encoder_embedding_block: TransformerEmbeddingBlock,
			decoder_embedding_block: TransformerEmbeddingBlock,
			encoder_block: DecoderBlock,
			decoder_block: Lass3DecoderBlock,
			collapse_block: CollapseBlock,
	):
		self.args = {
			'block_size': block_size,
			'encoder_embedding_block': encoder_embedding_block,
			'decoder_embedding_block': decoder_embedding_block,
			'encoder_block': encoder_block,
			'decoder_block': decoder_block,
			'collapse_block': collapse_block
		}
		super().__init__(input_size=(None, 2, block_size), auto_build=False)
		self.encoder_embedding_block = encoder_embedding_block
		self.decoder_embedding_block = decoder_embedding_block
		self.encoder_block = encoder_block
		self.decoder_block = decoder_block
		self.collapse_block = collapse_block

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x_encoder, x_decoder = x[:, 0, :], x[:, 1, :]

		x_encoder_embedded = self.encoder_embedding_block(x_encoder)
		y_encoder = self.encoder_block(x_encoder_embedded)

		x_decoder_embedded = self.decoder_embedding_block(x_decoder)
		y_decoder = self.decoder_block(y_encoder, x_decoder_embedded)

		y = self.collapse_block(y_encoder, y_decoder)
		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
