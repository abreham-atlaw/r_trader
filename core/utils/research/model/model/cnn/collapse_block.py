import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import FlattenLayer
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CollapseBlock(SpinozaModule):

	def __init__(
		self,
		ff_block: LinearModel = None,
		dropout: float = 0,
		extra_mode: bool = True
	):
		super().__init__(auto_build=False)
		self.args = {
			'ff_block': ff_block,
			'dropout': dropout,
			"extra_mode": extra_mode
		}
		self.ff_block = ff_block
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.extra_mode = extra_mode
		self.flatten = FlattenLayer(1, 2)

	def call(self, x: torch.Tensor, extra: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
		flattened = self.flatten(x)
		flattened = flattened.reshape(flattened.size(0), -1)
		flattened = self.dropout(flattened)

		concat = flattened
		if self.extra_mode:
			concat = torch.cat((concat, extra), dim=1)

		out = self.ff_block(concat)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
