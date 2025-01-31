import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class EDFDecoder(SpinozaModule):

	def __init__(
			self,
			channel_block: nn.Module,
			ff_block: nn.Module
	):
		super().__init__()
		self.args = {
			"channel_block": channel_block,
			"ff_block": ff_block
		}
		self.channel_block = channel_block
		self.ff_block = ff_block

	def call(self, y: torch.Tensor) -> torch.Tensor:
		out = self.channel_block(y)
		out = torch.flatten(out, start_dim=1, end_dim=2)
		out = self.ff_block(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
