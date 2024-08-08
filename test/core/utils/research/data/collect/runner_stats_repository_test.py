import typing
import unittest

import matplotlib.pyplot as plt

from core.di import ServiceProvider
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats


class RunnerStatsRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.repository = RunnerStatsRepository(ServiceProvider.provide_mongo_client())

	def __get_valid_dps(self) -> typing.List[RunnerStats]:
		dps = self.repository.retrieve_all()
		return [
			dp
			for dp in dps
			if dp.profit != 0
		]

	def test_plot_profit_vs_loss(self):
		dps = self.__get_valid_dps()
		print(f"Using {len(dps)} dps")
		self.assertGreater(len(dps), 0)

		plt.scatter(
			[dp.model_loss for dp in dps],
			[dp.profit for dp in dps]
		)
		plt.show()
