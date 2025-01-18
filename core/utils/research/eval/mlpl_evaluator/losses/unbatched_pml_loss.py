import torch
import torch.nn as nn

from core.utils.research.losses import ProximalMaskedLoss


class UnbatchedProximalMaskedLoss(ProximalMaskedLoss):

	def collapse(self, loss: torch.Tensor) -> torch.Tensor:
		return loss
