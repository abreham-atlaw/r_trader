import typing

import numpy as np

from core.di import ServiceProvider
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository


class RunnerStatsDataPreparer:

	def __init__(self, bounds=(-5, 5)):
		self.__repository = RunnerStatsRepository(
			ServiceProvider.provide_mongo_client()
		)
		self.__bounds = bounds

	def prepare(self) -> typing.Tuple[np.ndarray, np.ndarray]:
		stats = list(filter(
			lambda dp: dp.duration > 0 and 0 not in dp.model_losses and self.__bounds[0] <= dp.profit <= self.__bounds[1],
			self.__repository.retrieve_all()
		))

		X = np.stack([
			dp.model_losses
			for dp in stats
		])

		y = np.array([dp.profit for dp in stats])

		return X, y
