import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CollapseFFBlock(SpinozaModule):

	def __init__(
			self,
			target_channels: int,
			ff_block: LinearModel = None,
	):
		super(CollapseFFBlock, self).__init__()
		self.args = {
			'target_channels': target_channels,
			'ff_block': ff_block
		}
		self.target_channels = target_channels
		self.ff_block = ff_block

	def collapse(self, out: torch.Tensor) -> torch.Tensor:
		return torch.reshape(
			out,
			(out.shape[0], self.target_channels, -1)
		)

	def call(self, x, extra):
		collapsed = self.collapse(x)
		concatenated = torch.cat((collapsed, extra), dim=2)
		out = self.ff_block(concatenated)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
