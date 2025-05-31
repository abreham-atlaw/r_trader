import torch
import torch.nn as nn


class MinMaxNorm(nn.Module):

	def forward(self, x: torch.Tensor):
		x = x - torch.min(x, dim=-1, keepdim=True).values
		return x/torch.max(x, dim=-1, keepdim=True).values
