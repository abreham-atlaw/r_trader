import torch
from torch import nn

from core.utils.research.model.layers import MovingAverage


class WrappedModel(nn.Module):

	def __init__(self, model: nn.Module, window_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.ma = MovingAverage(window_size)

	def forward(self, inputs: torch.Tensor):
		inputs = self.ma(inputs)
		return self.model(inputs)
