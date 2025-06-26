import typing

import torch

from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.savable import SpinozaModule
from .transformer_block import TransformerBlock


class Transformer(SpinozaModule):

	def __init__(
			self,
			extra_len: int,
			transformer_block: TransformerBlock,
			collapse_block: CollapseBlock,
			input_size: int,
	):
		self.args = {
			'extra_len': extra_len,
			'transformer_block': transformer_block,
			'collapse_block': collapse_block,
			'input_size': input_size
		}
		super().__init__(auto_build=False, input_size=input_size)
		self.extra_len = extra_len
		self.transformer_block = transformer_block
		self.collapse_block = collapse_block
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		seq = x[:, :-self.extra_len]
		extra = x[:, -self.extra_len:]
		transformed = self.transformer_block(seq)
		out = self.collapse_block(transformed, extra=extra)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
