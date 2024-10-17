import torch.nn as nn

from core.utils.research.losses import PredictionConfidenceScore


class PredictionConfidenceScoreLoss(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__()
		self.score = PredictionConfidenceScore(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return 1/self.score(*args, **kwargs)
