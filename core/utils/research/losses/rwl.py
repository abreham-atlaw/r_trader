from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReverseWeightLoss(nn.Module, ABC):

	def __init__(self, softmax=True, device=None):
		super(ReverseWeightLoss, self).__init__()
		self.softmax = softmax
		if device is None:
			device = torch.device("cpu")
		self.device = device

	@abstractmethod
	def generate_weights(self, y: torch.Tensor) -> torch.Tensor:
		pass

	def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
		if self.softmax:
			y_hat = F.softmax(y_hat, dim=1)
		return torch.mean(torch.sum((1-self.generate_weights(y).to(self.device))*y_hat, dim=1))


class ReverseMAWeightLoss(ReverseWeightLoss):

	def __init__(self, window_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.window = window_size
		self.weights = torch.ones(1, 1, self.window).to(self.device)

	def generate_weights(self, s: torch.Tensor) -> torch.Tensor:
		print(s.shape)
		s = F.pad(s, (self.window - 1, self.window - 1))

		for _ in range(2):
			s = F.conv1d(s.unsqueeze(1),  self.weights/ self.window).squeeze(dim=1)

		s = s / torch.max(s, dim=1)[0]

		return s

