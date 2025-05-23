import typing

import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule


class LinearUSM(SpinozaModule):

	def __init__(
			self,
			models,
	):
		super().__init__(input_size=models[0].input_size, auto_build=False)
		self.args = {
			'models': models
		}
		self.models = models
		self.ff = nn.Linear(len(models), 1, bias=False)
		self.init()

	def call(self, x) -> torch.Tensor:
		with torch.no_grad():
			outs = torch.stack(
				[m(x).detach() for m in self.models],
				dim=2
			)

		return torch.squeeze(
			self.ff(
				outs
			),
			dim=2
		)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
