import typing
from abc import ABC, abstractmethod

import torch

from core.utils.research.model.model.savable import SpinozaModule


class MaskedStackedModel(SpinozaModule, ABC):

	def __init__(
			self,
			models: typing.List[SpinozaModule],
	):
		self.args = {
			"models": models
		}
		super().__init__(input_size=models[0].input_size, auto_build=False)
		self.models = models

	@abstractmethod
	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		pass

	def call(self, x: torch.Tensor) -> torch.Tensor:

		with torch.no_grad():
			y = torch.stack([
				m(x).detach() for m in self.models
			], dim=1)

		mask = self._get_mask(x, y)
		if mask.shape == y.shape[:-1]:
			mask = torch.repeat_interleave(
				torch.unsqueeze(
					mask,
					dim=2
				),
				y.shape[-1],
				dim=2
			)

		out = mask*y
		out = torch.sum(out, dim=1)

		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
