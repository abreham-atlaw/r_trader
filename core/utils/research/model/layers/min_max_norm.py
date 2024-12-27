import torch
import torch.nn as nn


class MinMaxNorm(nn.Module):

	def __init__(self, dim=-1):
		super(MinMaxNorm, self).__init__()
		self.dim = dim

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		inputs = inputs - torch.min(inputs, dim=self.dim, keepdim=True)[0]
		inputs = inputs / torch.max(inputs, dim=self.dim, keepdim=True)[0]
		return inputs
