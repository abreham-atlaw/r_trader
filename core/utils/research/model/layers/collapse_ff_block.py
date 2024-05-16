import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SavableModule


class CollapseFFBlock(SavableModule):

	def __init__(
			self,
			num_classes: int,
			input_channels: int,
			extra_len: int,
			ff_linear: LinearModel = None,
			linear_collapse=False,
			input_size: int = 1028,

	):
		super(CollapseFFBlock, self).__init__()
		self.args = {
			'ff_linear': ff_linear,
			'num_classes': num_classes,
			'linear_collapse': linear_collapse,
			'extra_len': extra_len,
			'input_size': input_size,
		}
		self.layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()
		self.input_size = input_size
		self.extra_len = extra_len
		self.input_channels = input_channels

		if ff_linear is None:
			self.fc = nn.Linear(input_channels+self.extra_len, num_classes)
		else:
			self.fc = nn.Sequential(
				nn.Linear(input_channels + self.extra_len, ff_linear.input_size),
				ff_linear,
				nn.Linear(ff_linear.output_size, num_classes)
			)

		self.collapse_layer = None if linear_collapse else nn.AdaptiveAvgPool1d((1,))

	def collapse(self, out: torch.Tensor) -> torch.Tensor:
		if self.collapse_layer is None:
			self.collapse_layer = nn.Linear(out.shape[-1], 1)
		return self.collapse_layer(out)

	def forward(self, x, extra):
		out = self.collapse(x)
		out = out.reshape(out.size(0), -1)
		out = torch.cat((out, extra), dim=1)
		out = self.fc(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
