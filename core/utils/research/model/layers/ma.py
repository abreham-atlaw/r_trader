import torch
import torch.nn as nn


class MovingAverage(nn.Module):

	def __init__(self, window_size):
		super(MovingAverage, self).__init__()
		self.window_size = window_size
		self.conv = nn.Conv1d(1, 1, self.window_size, bias=False)
		self.conv.weight.data.fill_(1.0 / self.window_size)

	def forward(self, x):
		x = x.unsqueeze(1)
		x = self.conv(x)
		return x.squeeze(1)
