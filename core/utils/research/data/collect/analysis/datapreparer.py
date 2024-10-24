import typing
from copy import deepcopy

import numpy as np

from core.di import ServiceProvider, ResearchProvider
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository


class RunnerStatsDataPreparer:

	def __init__(
			self,
			bounds=(-5, 5),
			columns=None,
			min_sessions=1
	):
		self.__repository = ResearchProvider.provide_runner_stats_repository()
		self.__bounds = bounds
		self.__columns = columns
		self.__min_sessions = min_sessions

	def __fetch_stats(
			self,
			loss_evaluated=True,
			runlive_tested=True,
			min_sessions=None,
	) -> typing.List[RunnerStats]:
		if min_sessions is None:
			min_sessions = self.__min_sessions

		stats: typing.List[RunnerStats] = self.__repository.retrieve_all()

		if min_sessions is not None:
			stats = list(filter(
				lambda dp: len(dp.session_timestamps) >= min_sessions,
				stats
			))

		if loss_evaluated:
			stats = list(filter(
				lambda dp: 0 not in dp.model_losses,
				stats
			))

		if runlive_tested:
			stats = list(filter(
				lambda dp: dp.duration > 0,
				stats
			))

		stats = list(filter(
			lambda dp: (
					self.__bounds[0] <= dp.profit <= self.__bounds[1]
			),
			stats
		))
		return stats

	def prepare(
			self,
			stats: typing.List[RunnerStats] = None,
			loss_evaluated=True,
			runlive_tested=True,
			min_sessions=None
	) -> typing.Tuple[np.ndarray, np.ndarray]:

		if stats is None:
			stats = self.__fetch_stats(
				loss_evaluated=loss_evaluated,
				runlive_tested=runlive_tested,
				min_sessions=min_sessions
			)

		if self.__columns is not None:
			stats = [
				deepcopy(dp)
				for dp in stats
			]
			for dp in stats:
				dp.model_losses = tuple([dp.model_losses[i] for i in self.__columns if i < len(dp.model_losses)])

		print(f"Preparing {len(stats)} stats")
		X = np.stack([
			dp.model_losses
			for dp in stats
		])
		X = X / np.max(X, axis=0)

		y = np.array([dp.profit for dp in stats])

		return X, y
