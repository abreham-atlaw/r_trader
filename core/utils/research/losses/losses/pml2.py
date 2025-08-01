import torch

from .pml import ProximalMaskedLoss


class ProximalMaskedLoss2(ProximalMaskedLoss):

	def __init__(self, *args, w: float = 1.0, h: float = 5.0, b: float = 1e-3, **kwargs):
		self.w = w
		self.h = h
		self.b = b
		super().__init__(*args, **kwargs)

	def _f(self, i: int) -> torch.Tensor:
		x = torch.arange(self.n)
		return ((1/(1 + torch.exp(self.w * torch.abs(x-i) - self.h))) + self.b) * (1/(1+self.b))
