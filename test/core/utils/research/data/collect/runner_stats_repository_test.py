import random
import typing
import unittest

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from pprint import pprint

import pandas as pd

from core.di import ServiceProvider
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats
from core.utils.research.data.collect.runner_stats_serializer import RunnerStatsSerializer


class RunnerStatsRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.repository = RunnerStatsRepository(ServiceProvider.provide_mongo_client())
		self.serializer = RunnerStatsSerializer()

	def __create_for_runlive(self) -> typing.List[RunnerStats]:
		stats = []
		for i in range(10):
			stat = RunnerStats(
				id=str(i),
				model_name="test",
				profit=0.0,
				duration=0.0,
				model_losses=(3.0, 7.0),
				session_timestamps=[datetime(year=2020, month=1, day=1)]
			)
			stat.add_duration(i*60*60)
			self.repository.store(stat)
			stats.append(stat)
		return stats

	def __get_valid_dps(self) -> typing.List[RunnerStats]:
		dps = self.repository.retrieve_all()
		return [
			dp
			for dp in dps
			if dp.profit != 0 and 0 not in dp.model_losses
		]

	def __print_dps(self, dps: typing.List[RunnerStats]):
		print(pd.DataFrame([
			(dp.id, dp.model_name, dp.profit, dp.model_losses, dp.session_timestamps, dp.duration)
			for dp in dps
		], columns=["ID", "Model", "Profit", "Losses", "Sessions", "Duration"]).to_string())

	def test_plot_profit_vs_loss(self):
		dps = self.__get_valid_dps()
		print(f"Using {len(dps)} dps")
		self.assertGreater(len(dps), 0)
		losses = [
			[dp.model_losses[i] for dp in dps]
			for i in range(2)
		]
		losses.append(
			[np.prod(dp.model_losses) for dp in dps]
		)
		for i in range(3):
			plt.figure()
			plt.scatter(
				losses[i],
				[dp.profit for dp in dps]
			)
		plt.show()

	def test_plot_losses(self):
		dps = self.repository.retrieve_by_loss_complete()
		print(f"Using {len(dps)} dps")
		plt.scatter(
			*[
				[dp.model_losses[i] for dp in dps]
				for i in range(2)
			]
		)
		plt.show()

	def test_get_all(self):
		stats = self.repository.retrieve_all()
		self.assertGreater(len(stats), 0)

	def test_store(self):

		ID = "test_id"
		stat = RunnerStats(
			id=ID,
			model_name="test",
			profit=0.0,
			duration=0.0,
			model_losses=(3.0, 7.0),
			session_timestamps=[datetime(year=2020, month=1, day=1)]
		)

		self.repository.store(stat)

		retrieved_stat = self.repository.retrieve(ID)

		self.assertEqual(self.serializer.serialize(stat), self.serializer.serialize(retrieved_stat))

		self.repository.remove(ID)

	def test_allocate(self):
		created = self.__create_for_runlive()
		allocated = []
		for i in range(len(created)):
			stat = self.repository.allocate_for_runlive()
			self.assertNotIn(stat.id, [stat.id for stat in allocated])
			allocated.append(stat)
			print(f"Allocated {len(allocated)} of {len(created)}")

		print("Allocated:", [stat.duration for stat in allocated])
		for stat in allocated:
			self.repository.finish_session(stat, random.random())

		for stat in allocated:
			retrieved = self.repository.retrieve(stat.id)
			self.assertNotEqual(retrieved.profit, 0)

	def test_get_completed_loss_percentage(self):
		all = self.repository.retrieve_all()
		completed = [stat for stat in all if stat.model_losses[1] != 0]

		print(f"All: {len(all)}")
		print(f"Completed: {len(completed)}")
		print(f"Percentage: {len(completed) / len(all)}")

	def test_get_least_loss_losing_stats(self):
		dps = sorted([
			dp
			for dp in self.__get_valid_dps()
			if dp.profit < 0
		], key=lambda dp: dp.model_losses[1]
		)

		self.__print_dps(dps)

	def test_get_sessions(self):
		dps = sorted(
			self.__get_valid_dps(),
			key=lambda dp: dp.session_timestamps[-1]
		)

		self.__print_dps(dps)

