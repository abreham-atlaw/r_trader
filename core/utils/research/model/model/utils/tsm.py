import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class TemperatureScalingModel(nn.Module):

	def __init__(self, model: nn.Module, temperature: float, *args, **kwargs):
		super().__init__(
			*args,
			**kwargs
		)
		self.model = model
		self.temperature = temperature

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		return self.model(X) / self.temperature

	def __getattribute__(self, item):
		# Avoid infinite recursion by directly accessing attributes in self.__dict__
		if item in ('model', 'temperature', '__dict__', '_modules'):
			return super().__getattribute__(item)
		try:
			return super().__getattribute__(item)
		except AttributeError:
			return getattr(self.model, item)
