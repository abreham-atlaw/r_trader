import torch
from torch import nn


class ProximalMaskedLoss(nn.Module):

	def __init__(self, n: int, *args, p: int = 1, softmax=True, epsilon=1e-9, device='cpu', **kwargs):
		super().__init__(*args, **kwargs)
		self.n = n
		self.p = p
		self.activation = nn.Softmax() if softmax else nn.Identity()
		self.mask = self.__generate_mask(n, p).to(device)
		self.epsilon = epsilon
		self.device = device

	@staticmethod
	def __f(t, n, p):
		return (1 / (torch.abs(torch.arange(n) - t) + 1)) ** p

	@staticmethod
	def __generate_mask(n, p) -> torch.Tensor:

		return torch.stack([
			ProximalMaskedLoss.__f(t, n, p) for t in range(n)
		])

	def forward(self, y, y_hat) -> torch.Tensor:
		y_hat = self.activation(y_hat)

		y_mask = torch.sum(
			self.mask * torch.unsqueeze(y, dim=2),
			dim=1
		)
		return torch.mean(
			(1/(torch.sum(y_mask * y_hat, dim=1) - self.epsilon)) - 1
		)
