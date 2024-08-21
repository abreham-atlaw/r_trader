import typing

import torch
import torch.nn as nn
import numpy as np

from .msce import MeanSquaredClassError


class MutliLoss(nn.Module):

	def __init__(self, losses: typing.List[nn.Module]):
		super().__init__()
		self.losses = nn.ModuleList(losses)

	def forward(self, *args, **kwargs):
		return sum([loss(*args, **kwargs) for loss in self.losses])


class MSCECrossEntropyLoss(MutliLoss):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor], epsilon: float = None, device=None):
		super().__init__([MeanSquaredClassError(classes, epsilon, device), nn.CrossEntropyLoss()])
