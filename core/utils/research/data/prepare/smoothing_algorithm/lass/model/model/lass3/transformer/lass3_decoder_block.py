import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import DecoderBlock
from .cross_attention_block import CrossAttentionBlock


class Lass3DecoderBlock(SpinozaModule):

	def __init__(
			self,
			self_attention_block: DecoderBlock,
			cross_attention_block: CrossAttentionBlock
	):
		super().__init__()
		self.args = {
			'self_attention_block': self_attention_block,
			'cross_attention_block': cross_attention_block
		}
		self.self_attention_block = self_attention_block
		self.cross_attention_block = cross_attention_block

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:

		self_attention = self.self_attention_block(x_decoder)
		out = self.cross_attention_block(x_encoder, self_attention)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
