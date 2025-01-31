import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CollapseBlock(SpinozaModule):

	def __init__(
		self,
		ff_block: LinearModel = None,
		dropout: float = 0,
	):
		super().__init__(auto_build=False)
		self.args = {
			'ff_block': ff_block,
			'dropout': dropout,
		}
		self.ff_block = ff_block
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

	def call(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
		out = torch.flatten(x, 1, 2)
		out = out.reshape(out.size(0), -1)
		out = self.dropout(out)
		out = torch.cat((out, extra), dim=1)
		out = self.ff_block(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
