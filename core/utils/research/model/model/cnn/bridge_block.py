import typing

import torch

from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class BridgeBlock(SpinozaModule):

	def __init__(
			self,
			ff_block: LinearModel,
	):
		super().__init__(auto_build=False)
		self.args = {
			"ffn": ff_block
		}
		self.ffn = ff_block

	def call(self, x: torch.Tensor) -> torch.Tensor:
		out = self.ffn(x)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
