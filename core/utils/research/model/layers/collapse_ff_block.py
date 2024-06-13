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
			input_shape: typing.Optional[typing.Tuple[int, int]] = None,
	):
		super(CollapseFFBlock, self).__init__()
		self.args = {
			'ff_linear': ff_linear,
			'num_classes': num_classes,
			'linear_collapse': linear_collapse,
			'extra_len': extra_len,
			'input_shape': input_shape,
			'input_channels': input_channels
		}
		self.input_shape = input_shape
		self.layers = nn.ModuleList()
		self.pool_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()
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
		self.num_classes = num_classes
		self.ff_linear = ff_linear
		self.fc_layer = None
		self.collapse_layer = None if linear_collapse else nn.AdaptiveAvgPool1d((1,))
		if input_shape is not None:
			self.__init()

	def __init(self):
		init_data = torch.rand((1, *self.input_shape))
		extra = torch.rand((1, self.extra_len))
		self(init_data, extra)

	def collapse(self, out: torch.Tensor) -> torch.Tensor:
		return torch.flatten(out, 1, 2)

	def fc(self, out: torch.Tensor) -> torch.Tensor:
		if self.fc_layer is None:
			if self.ff_linear is None:
				self.fc_layer = nn.Linear(out.shape[-1], self.num_classes)
			else:
				self.fc_layer = nn.Sequential(
					nn.Linear(out.shape[-1], self.ff_linear.input_size),
					self.ff_linear,
					nn.Linear(self.ff_linear.output_size, self.num_classes)
				)
		return self.fc_layer(out)

	def forward(self, x, extra):
		out = self.collapse(x)
		out = out.reshape(out.size(0), -1)
		out = torch.cat((out, extra), dim=1)
		out = self.fc(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		self.args["input_shape"] = self.input_shape
		return self.args
