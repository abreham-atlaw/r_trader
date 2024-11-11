import typing
from abc import ABC, abstractmethod

import torch
from torch import nn

from core.utils.research.training.trackers.stats_tracker import StatsTracker
from core.utils.research.training.trackers.tracker import TorchTracker


class LambdaStatsTracker(StatsTracker):

	def __init__(self, callback: typing.Callable, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__callback = callback

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
		return self.__callback(X, y, y_hat, model, loss, gradient, epoch, batch)


class ListStatsTracker(TorchTracker, ABC):

	def __init__(self, label: str, *args, **kwargs):
		super().__init__()
		self.__trackers = None
		self.__args, self.__kwargs = args, kwargs
		self.__label = label

	def __init_trackers(self, length):
		self.__trackers = [
			LambdaStatsTracker(
				*self.__args,
				**self.__kwargs,
				label=f"{self.__label}_{i}",
				callback=lambda *args, **kwargs: self._extract_list_values(*args, **kwargs)[i],
			)
			for i in range(length)
		]

	@abstractmethod
	def _extract_list_values(
			self,
			X: torch.Tensor,
			y: torch.Tensor,
			y_hat: torch.Tensor,
			model: nn.Module,
			loss: torch.Tensor,
			gradient: torch.Tensor,
			epoch: int,
			batch: int
	) -> typing.List[torch.Tensor]:
		pass

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
		values = self._extract_list_values(X, y, y_hat, model, loss, gradient, epoch, batch)
		if self.__trackers is None:
			self.__init_trackers(len(values))
		for tracker, value in zip(self.__trackers, values):
			tracker.on_batch_end(X, y, y_hat, model, loss, gradient, epoch, batch)


class WeightsStatsTracker(ListStatsTracker):

	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			label="weights",
			**kwargs
		)

	def _extract_list_values(
			self,
			X: torch.Tensor,
			y: torch.Tensor,
			y_hat: torch.Tensor,
			model: nn.Module,
			loss: torch.Tensor,
			gradient: torch.Tensor,
			epoch: int,
			batch: int
	) -> typing.List[torch.Tensor]:
		return list(model.parameters())


class GradientsStatsTracker(ListStatsTracker):

	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			label="gradients",
			**kwargs
		)

	def _extract_list_values(
			self,
			X: torch.Tensor,
			y: torch.Tensor,
			y_hat: torch.Tensor,
			model: nn.Module,
			loss: torch.Tensor,
			gradient: torch.Tensor,
			epoch: int,
			batch: int
	) -> typing.List[torch.Tensor]:

		return gradient
