import torch
import torch.nn as nn


class MovingAverage(nn.Module):

	def __init__(self, window_size):
		super(MovingAverage, self).__init__()
		self.window_size = window_size
		self.avg_pool = nn.AvgPool1d(kernel_size=window_size, stride=1)

	def forward(self, x):
		return self.avg_pool(x)
