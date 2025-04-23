import typing

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from core.utils.research.losses import ClassPerformanceLoss


class ClassPerformanceEvaluator:

	def __init__(
			self,
			loss: ClassPerformanceLoss,
			dataloader: DataLoader,
			model_class_range: typing.Optional[typing.Tuple[int, int]] = None
	):

		self.loss = loss
		self.dataloader = dataloader
		self.model_class_range = model_class_range

	def evaluate(self, model: nn.Module):

		model.eval()

		running_loss = None
		running_size = 0

		for x, y in self.dataloader:
			with torch.no_grad():
				y_hat = model(x)

			if self.model_class_range is not None:
				y_hat = y_hat[:, self.model_class_range[0]:self.model_class_range[1]]
				y = y[:, self.model_class_range[0]:self.model_class_range[1]]

			loss = self.loss(y_hat, y)

			running_size += y_hat.shape[0]
			loss *= y_hat.shape[0]
			if running_loss is None:
				running_loss = torch.unsqueeze(loss, dim=0)
				continue
			running_loss = torch.cat((running_loss, torch.unsqueeze(loss, dim=0)), dim=0)

		return torch.sum(running_loss, dim=0) / running_size
