import torch

from .pml import ProximalMaskedLoss


class ProximalMaskedPenaltyLoss(ProximalMaskedLoss):

	def _generate_mask(self) -> torch.Tensor:
		return 1-super()._generate_mask()

	def _loss(self, y_mask: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
		return torch.sum(y_mask * y_hat, dim=1)
