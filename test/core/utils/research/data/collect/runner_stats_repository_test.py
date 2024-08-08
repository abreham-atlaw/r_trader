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
			[dp.model_losses[0] for dp in dps],
			[dp.profit for dp in dps]
		)
		plt.show()

	def test_get_all(self):
		stats = self.repository.retrieve_all()
		self.assertGreater(len(stats), 0)

	def test_create(self):
		ID = "test_id"
		stat = RunnerStats(
			id=ID,
			model_name="test",
			profit=0.0,
			duration=0.0,
			model_losses=(0.0, 0.0)
		)

		self.repository.store(stat)

		retrieved_stat = self.repository.retrieve(ID)
		self.assertEqual(stat, retrieved_stat)
