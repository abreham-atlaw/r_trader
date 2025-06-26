import typing

import torch
from torch import nn

from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import TransformerBlock


class BridgeBlock(SpinozaModule):

	def __init__(
			self,
			ff_block: LinearModel = None,
			transformer_block: TransformerBlock = None
	):
		super().__init__(auto_build=False)
		self.args = {
			"ff_block": ff_block,
			"transformer_block": transformer_block
		}
		self.ffn = ff_block if ff_block is not None else nn.Identity()
		self.transformer_block = transformer_block if transformer_block is not None else nn.Identity()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		out = self.transformer_block(x)
		out = self.ffn(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
