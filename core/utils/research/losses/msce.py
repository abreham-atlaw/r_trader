import typing

import numpy as np
import torch
import torch.nn as nn


class MeanSquaredClassError(nn.Module):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor]):
		super().__init__()
		if isinstance(classes, np.ndarray):
			classes = torch.from_numpy(classes)
		self.classes = classes

	def get_class_value(self, y: torch.Tensor) -> torch.Tensor:
		return torch.sum(y * self.classes, dim=1)

	def forward(self, y, y_hat) -> torch.Tensor:
		loss = (self.get_class_value(y) - self.get_class_value(y_hat))**2
		return torch.mean(loss)
