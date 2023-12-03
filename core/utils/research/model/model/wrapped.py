import torch
from torch import nn

from core.utils.research.model.layers import MovingAverage


class WrappedModel(nn.Module):

	def __init__(self, model: nn.Module, window_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.ma = MovingAverage(window_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, inputs: torch.Tensor):
		inputs = self.ma(inputs)
		outputs = self.model(inputs)
		return self.softmax(outputs)
