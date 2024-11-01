import torch
from torch import nn

from .stats_tracker import StatsTracker


class Keys:

	X = "X"
	Y = "y"
	YHAT = "y_hat"
	LOSS = "loss"

	ALL = X, Y, YHAT, LOSS


class DynamicStatsTracker(StatsTracker):

	def __init__(self, model_name: str, label: str):
		super().__init__(model_name, label)
		self.__key = label

	def _extract_values(
			self,
			X: torch.Tensor,
			y: torch.Tensor,
			y_hat: torch.Tensor,
			model: nn.Module,
			loss: torch.Tensor,
			gradient: torch.Tensor,
			epoch: int,
			batch: int
	) -> torch.Tensor:

		return {
			Keys.X: X,
			Keys.Y: y,
			Keys.YHAT: y_hat,
			Keys.LOSS: torch.unsqueeze(loss, dim=0),
		}.get(self.__key)
