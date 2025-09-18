import torch
from torch import nn


class SmoothedChannelDropout(nn.Module):

	def __init__(self, depth_dropout: float, batch_dropout: float, channel: int = 1):
		super(SmoothedChannelDropout, self).__init__()
		self.depth_dropout = depth_dropout
		self.batch_dropout = batch_dropout
		self.channel = channel

	def dropout(self, x: torch.Tensor) -> torch.Tensor:
		probs = torch.rand(x.shape[0])*self.depth_dropout
		lengths = probs * x.shape[-1]
		lengths = lengths.int()

		mask = torch.arange(x.size(-1)).unsqueeze(0) < lengths.unsqueeze(1)
		mask = mask.to(x.device)
		x[:, 1, :][mask] = 0

		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if (not self.training) or self.depth_dropout == 0 or self.batch_dropout == 0:
			return x

		idxs = torch.rand(x.shape[0]) < self.batch_dropout
		x[idxs] = self.dropout(x[idxs])
		return x
