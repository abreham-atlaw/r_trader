import typing

import torch
import numpy as np

from core.utils.research.losses import MeanSquaredClassError
from core.utils.research.losses.spinoza_loss import SpinozaLoss


class OutputBatchClassVarianceScore(SpinozaLoss):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor], epsilon: float = None, device=None, softmax=True):
		super(OutputBatchClassVarianceScore, self).__init__()
		self.__loss = MeanSquaredClassError(
			classes=classes,
			epsilon=epsilon,
			device=device,
			softmax=softmax
		)

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		classes = self.__loss.get_class_value(y_hat)
		variance = torch.var(classes, dim=0)
		return variance
