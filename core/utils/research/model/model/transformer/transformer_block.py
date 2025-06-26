import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from .decoder_block import DecoderBlock


class TransformerBlock(SpinozaModule):

	def __init__(
			self,
			decoder_block: DecoderBlock,
			encoder_block: DecoderBlock
	):
		self.args = {
			'decoder_block': decoder_block,
			'encoder_block': encoder_block
		}
		super().__init__()
		self.decoder_block = decoder_block
		self.encoder_block = encoder_block

	def call(self, x: torch.Tensor) -> torch.Tensor:
		decoded = self.decoder_block(x)
		encoded = self.encoder_block(decoded)
		return encoded

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
