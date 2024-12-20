import typing

import torch
import torch.nn as nn

from .masked_stacked_model import MaskedStackedModel
from ...savable import SpinozaModule


class PlainMSM(MaskedStackedModel):

	def __init__(self, models: typing.List[SpinozaModule]):
		super().__init__(models=models)
		self.w = nn.Parameter(torch.ones((len(models,)), dtype=torch.float32))
		self.init()

	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return torch.ones((y.shape[0], self.w.shape[0]), dtype=torch.float32, device=y.device) * self.w
