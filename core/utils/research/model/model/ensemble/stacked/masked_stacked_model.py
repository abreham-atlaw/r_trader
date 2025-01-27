import typing
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class MaskedStackedModel(SpinozaModule, ABC):

	def __init__(
			self,
			models: typing.List[SpinozaModule],
			pre_weight_softmax: bool = False
	):
		self.args = {
			"models": models,
			"pre_weight_softmax": pre_weight_softmax
		}
		super().__init__(input_size=models[0].input_size, auto_build=False)
		self.models = nn.ModuleList(models)
		self.softmax = nn.Softmax(dim=-1) if pre_weight_softmax else nn.Identity()

	@abstractmethod
	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		pass

	def _get_model_outputs(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			y = torch.stack([
				m(x).detach() for m in self.models
			], dim=1)
		return y

	def _apply_mask(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

	def get_and_apply_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		y = self.softmax(y)
		mask = self._get_mask(x, y)
		out = self._apply_mask(y, mask)
		return out

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y = self._get_model_outputs(x)
		out = self.get_and_apply_mask(x, y)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
