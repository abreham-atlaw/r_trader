from abc import ABC, abstractmethod

import torch
from torch import nn

from core.di import ResearchProvider
from core.utils.research.training.trackers.tracker import TorchTracker


class StatsTracker(TorchTracker, ABC):

	def __init__(
			self,
			model_name: str,
			label: str
	):
		super().__init__()
		self.__repository = ResearchProvider.provide_ims_repository(
			model_name, label
		)

	@abstractmethod
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
		pass

	def on_batch_end(
			self,
			X: torch.Tensor,
			y: torch.Tensor,
			y_hat: torch.Tensor,
			model: nn.Module,
			loss: torch.Tensor,
			gradient: torch.Tensor,
			epoch: int,
			batch: int
	):
		value = self._extract_values(
			X=X,
			y=y,
			y_hat=y_hat,
			model=model,
			loss=loss,
			gradient=gradient,
			epoch=epoch,
			batch=batch
		)
		self.__repository.store(value, epoch, batch)
