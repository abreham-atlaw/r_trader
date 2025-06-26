import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from .teb import TransformerEmbeddingBlock
from .decoder_block import DecoderBlock


class TransformerBlock(SpinozaModule):

	def __init__(
			self,
			transformer_embedding_block: TransformerEmbeddingBlock,
			decoder_block: DecoderBlock,
			encoder_block: DecoderBlock,
			embedding_last: bool = False
	):
		self.args = {
			'transformer_embedding_block': transformer_embedding_block,
			'decoder_block': decoder_block,
			'encoder_block': encoder_block,
			'embedding_last': embedding_last
		}
		super().__init__()
		self.embedding_block = transformer_embedding_block
		self.decoder_block = decoder_block
		self.encoder_block = encoder_block
		self.embedding_last = embedding_last

	def call(self, x: torch.Tensor) -> torch.Tensor:
		embedded = self.embedding_block(x)

		decoded = self.decoder_block(embedded)
		encoded = self.encoder_block(decoded)

		if not self.embedding_last:
			encoded = torch.transpose(encoded, 1, 2)

		return encoded

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
