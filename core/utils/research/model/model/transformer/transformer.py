import typing

import torch

from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer.decoder_block import DecoderBlock


class Transformer(SpinozaModule):

	def __init__(
			self,
			extra_len: int,
			decoder_block: DecoderBlock,
			collapse_block: CollapseBlock,
			input_size: int,
	):
		self.args = {
			'extra_len': extra_len,
			'decoder_block': decoder_block,
			'collapse_block': collapse_block,
			'input_size': input_size
		}
		super().__init__(auto_build=False, input_size=input_size)
		self.extra_len = extra_len
		self.decoder_block = decoder_block
		self.collapse_block = collapse_block
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		seq = x[:, :-self.extra_len]
		extra = x[:, -self.extra_len:]
		decoded = self.decoder_block(seq)
		out = self.collapse_block(decoded, extra=extra)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
