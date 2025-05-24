import torch
import torch.nn as nn


class LogLoss(nn.Module):

	def __init__(self, loss_fn: nn.Module, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.loss_fn = loss_fn

	def forward(self, *args, **kwargs):
		return torch.log(
			self.loss_fn(*args, **kwargs)
		)

