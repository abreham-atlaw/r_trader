import torch
import torch.nn as nn

from core.utils.research.losses import OutputBatchVariance


class OutputBatchVarianceLoss(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__()
		self.loss = OutputBatchVariance(*args, **kwargs)

	def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
		return 1/self.loss(y_hat)
