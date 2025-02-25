
import torch
import torch.nn as nn


class AxisFFN(nn.Module):

	def __init__(self, ffn: nn.Module, axis: int):
		super(AxisFFN, self).__init__()
		self.ffn = ffn
		self.axis = axis

	def forward(self, x):
		x = x.transpose(self.axis, -1)
		out = self.ffn(x)
		return out.transpose(self.axis, -1)
