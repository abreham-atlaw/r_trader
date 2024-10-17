import torch.nn as nn


class ScoreLoss(nn.Module):

	def __init__(self, loss: nn.Module):
		super().__init__()
		self.loss = loss

	def forward(self, *args, **kwargs):
		return 1/self.loss(*args, **kwargs)
