import typing

import torch
import torch.nn as nn
import numpy as np

from .msce import MeanSquaredClassError


class MultiLoss(nn.Module):

	def __init__(self, losses: typing.List[nn.Module], weights: typing.Union[typing.List[float], np.ndarray, torch.Tensor] = None):
		super().__init__()
		self.losses = nn.ModuleList(losses)
		if weights is None:
			weights = np.ones(len(losses))
		if isinstance(weights, list):
			weights = torch.Tensor(weights)
		if isinstance(weights, np.ndarray):
			weights = torch.from_numpy(weights)
		self.weights = weights

	def forward(self, *args, **kwargs):
		return torch.sum(
			torch.Tensor([loss(*args, **kwargs) for loss in self.losses]) * self.weights
		)


class MSCECrossEntropyLoss(MultiLoss):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor], epsilon: float = None, device=None, *args, **kwargs):
		super().__init__([MeanSquaredClassError(classes, epsilon, device), nn.CrossEntropyLoss()], *args, **kwargs)
