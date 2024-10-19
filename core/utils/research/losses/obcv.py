import typing

import torch
import numpy as np
from torch import nn

from core.utils.research.losses import MeanSquaredClassError


class OutputBatchClassVariance(nn.Module):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor], epsilon: float = None, device=None, softmax=True):
		super(OutputBatchClassVariance, self).__init__()
		self.__loss = MeanSquaredClassError(
			classes=classes,
			epsilon=epsilon,
			device=device,
			softmax=softmax
		)

	def forward(self, y_hat, y):
		classes = self.__loss.get_class_value(y_hat)
		variance = torch.var(classes, dim=0).mean()
		return variance
