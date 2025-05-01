from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SpinozaLoss(nn.Module, ABC):

	def __init__(
			self,
			*args,
			weighted_sample: bool = False,
			collapsed: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.weighted_sample = weighted_sample
		self.collapsed = collapsed

	@abstractmethod
	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		pass

	def _collapse(self, loss: torch.Tensor) -> torch.Tensor:
		return torch.mean(loss)

	def _apply_weights(self, loss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
		return loss * w

	def forward(self, y_hat: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
		loss = self._call(y_hat, y)

		if self.weighted_sample:
			loss = self._apply_weights(loss, w)

		if self.collapsed:
			loss = self._collapse(loss)

		return loss
