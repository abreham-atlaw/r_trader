import typing
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from core.utils.research.model.layers import InverseSoftmax
from core.utils.research.model.model.savable import SpinozaModule


class MaskedStackedModel(SpinozaModule, ABC):

	def __init__(
			self,
			models: typing.List[SpinozaModule],
			extra_outputs_size: int = 1,
			vote: bool = False
	):
		self.args = {
			"models": models,
			"vote": vote,
			"extra_outputs_size": extra_outputs_size
		}
		super().__init__(input_size=models[0].input_size, auto_build=False)
		self.models = nn.ModuleList(models)
		self.softmax = nn.Softmax(dim=-1) if vote else nn.Identity()
		self.inverse_softmax = InverseSoftmax() if vote else nn.Identity()
		self.extra_outputs_size = extra_outputs_size
		self.vote = vote

	@abstractmethod
	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		pass

	def _get_model_outputs(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			y = torch.stack([
				m(x).detach() for m in self.models
			], dim=1)
		return y

	def _apply_to_prob(self, y: torch.Tensor, func: typing.Callable) -> torch.Tensor:
		original_shape = y.shape
		if y.dim() == 2:
			y = torch.unsqueeze(y, dim=1)
		return torch.reshape(
			torch.concatenate(
				(
					func(y[:, :, :-self.extra_outputs_size]),
					y[:, :, -self.extra_outputs_size:]
				),
				dim=2
			),
			original_shape
		)

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
		y = self._apply_to_prob(
			y,
			func=self.softmax
		)
		mask = self._get_mask(x, y)
		if self.vote and mask.dim() != 2:
			raise ValueError(f"Invalid mask shape: {mask.shape}, expected 2 dimensions when using vote.")

		mask = self.softmax(mask)
		out = self._apply_mask(y, mask)
		out = self._apply_to_prob(
			out,
			func=self.inverse_softmax
		)
		return out

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y = self._get_model_outputs(x)
		out = self.get_and_apply_mask(x, y)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
