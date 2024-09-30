import typing

import numpy as np
import torch
import torch.nn as nn


class MeanSquaredClassError(nn.Module):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor], epsilon: float = None, device=None, softmax=True):
		super().__init__()
		if isinstance(classes, list):
			classes = np.array(classes)
		if isinstance(classes, np.ndarray):
			classes = torch.from_numpy(classes)

		self.classes = classes
		if epsilon is not None:
			self.classes = torch.concatenate((self.classes, torch.Tensor([self.classes[-1] * (1+ epsilon)])))

		if device is not None:
			self.classes = self.classes.to(device)

		self.softmax = nn.Softmax() if softmax else nn.Identity()

	def get_class_value(self, y: torch.Tensor) -> torch.Tensor:
		return torch.sum(y * self.classes, dim=1)

	def forward(self, y, y_hat) -> torch.Tensor:
		y_hat = self.softmax(y_hat)
		loss = (self.get_class_value(y) - self.get_class_value(y_hat))**2
		return torch.mean(loss)
