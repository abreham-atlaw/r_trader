import torch
import torch.nn as nn

from core.utils.research.losses import SpinozaLoss


class CrossEntropyLoss(SpinozaLoss):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__loss = nn.CrossEntropyLoss()

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return self.__loss(y_hat, y)
