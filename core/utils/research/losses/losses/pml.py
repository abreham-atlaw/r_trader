import typing

import numpy as np
import torch
from torch import nn

from core.utils.research.losses import SpinozaLoss


class ProximalMaskedLoss(SpinozaLoss):

	def __init__(
			self,
			n: int,
			*args,
			p: int = 1,
			weights: typing.Optional[typing.Union[torch.Tensor, typing.List[float], np.ndarray]] = None,
			softmax=True,
			epsilon=1e-9,
			device='cpu',
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.n = n
		self.p = p
		self.activation = nn.Softmax(dim=1) if softmax else nn.Identity()
		self.mask = self.__generate_mask(n, p).to(device)
		self.epsilon = epsilon
		self.device = device

		if weights is None:
			weights = torch.ones(n)
		if isinstance(weights, list):
			weights = torch.Tensor(weights)
		if isinstance(weights, np.ndarray):
			weights = torch.from_numpy(weights)

		self.w = weights.to(device)

	@staticmethod
	def __f(t, n, p):
		return (1 / (torch.abs(torch.arange(n) - t) + 1)) ** p

	@staticmethod
	def __generate_mask(n, p) -> torch.Tensor:

		return torch.stack([
			ProximalMaskedLoss.__f(t, n, p) for t in range(n)
		])

	def collapse(self, loss: torch.Tensor) -> torch.Tensor:
		return torch.mean(loss)

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		y_hat = self.activation(y_hat)

		y_mask = torch.sum(
			self.mask * torch.unsqueeze(y, dim=2),
			dim=1
		)

		loss = (1 / (torch.sum(y_mask * y_hat, dim=1) - self.epsilon)) - 1
		w = torch.sum(self.w * y, dim=1)
		loss = loss*w

		return loss
