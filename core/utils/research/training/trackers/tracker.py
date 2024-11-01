import typing

import torch
from torch import nn


class TorchTracker:

	def on_batch_end(
			self,
			X: torch.Tensor,
			y: torch.Tensor,
			y_hat: torch.Tensor,
			model: nn.Module,
			loss: torch.Tensor,
			gradient: typing.List[torch.Tensor],
			epoch: int,
			batch: int
	):
		pass
