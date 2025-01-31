import typing
from abc import ABC, abstractmethod

import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule


class StackedEnsembleModel(SpinozaModule, ABC):

	def __init__(
			self,
			models: typing.List[nn.Module],
			*args, **kwargs
	):
		super().__init__(*args, input_size=models[0].input_size, output_size=models[0].output_size, auto_build=False, **kwargs)
		self.args = {
			"models": models
		}
		self.models = nn.ModuleList(models)

	@abstractmethod
	def _call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		pass

	def _get_model_outputs(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			y = torch.stack([
				m(x).detach() for m in self.models
			], dim=1)
		return y

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y = self._get_model_outputs(x)
		return self._call(x, y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
