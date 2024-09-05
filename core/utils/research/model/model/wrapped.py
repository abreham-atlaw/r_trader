import torch
from torch import nn

from core.utils.research.model.layers import MovingAverage


class WrappedModel(nn.Module):

	def __init__(self, model: nn.Module, seq_len: int, window_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.seq_len = seq_len
		self.ma = MovingAverage(window_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, inputs: torch.Tensor):
		mad = self.ma(inputs[:, :-1])
		inputs = torch.concat(
			[
				mad,
				inputs[:, -1:, :mad.shape[-1]]
			],
			dim=1
		)
		outputs = self.model(inputs)
		return torch.concat(
			[
				self.softmax(outputs[:, :-1]),
				outputs[:, -1:]
			],
			dim=1
		)
