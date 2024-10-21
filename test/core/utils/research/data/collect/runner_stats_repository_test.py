import random
import typing
import unittest

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
from pprint import pprint

import pandas as pd

from core.di import ServiceProvider, ResearchProvider
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository, RunnerStats
from core.utils.research.data.collect.runner_stats_serializer import RunnerStatsSerializer


class RunnerStatsRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.repository: RunnerStatsRepository = ResearchProvider.provide_runner_stats_repository()
		self.serializer = RunnerStatsSerializer()

		self.loss_names = [
			"nn.CrossEntropyLoss()",
			"ProximalMaskedLoss",
			"MeanSquaredClassError",
			"ReverseMAWeightLoss(window_size=10, softmax=True)",
			"PredictionConfidenceScore(softmax=True)",
			"OutputClassesVariance(softmax=True)",
			"OutputBatchVariance(softmax=True)",
			"OutputBatchClassVariance(softmax=True)",
		]

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
			if (dp.duration > 0) and (0 not in dp.model_losses)
		]

	def __print_dps(self, dps: typing.List[RunnerStats]):
		print(pd.DataFrame([
			(dp.id, dp.model_name, dp.duration, dp.profit, dp.real_profit,  dp.model_losses, dp.session_timestamps, dp.profits, dp.real_profits)
			for dp in dps
		], columns=["ID", "Model", "Duration", "Profit", "Real Profit", "Losses", "Sessions", "Profits", "Real Profits"]).to_string())

	def __filter_stats(
			self,
			dps: typing.List[RunnerStats],
			time: datetime = None,
			model_losses: typing.Tuple[float, float] = None,
			min_profit: float = None,
			max_profit: float = None,
			model_key: str = None,
			min_duration: float = None
	) -> typing.List[RunnerStats]:

		if model_key is not None:
			dps = [
				dp
				for dp in dps
				if model_key in dp.model_name
			]

		if time is not None:
			dps = [
				dp
				for dp in dps
				if dp.session_timestamps[-1] > time
			]
		if model_losses is not None:

			for i in range(len(model_losses)):
				if model_losses[i] is not None:
					dps = list(filter(
						lambda dp: dp.model_losses[i] < model_losses[i],
						dps
					))

		if min_profit is not None:
			dps = list(filter(
				lambda dp: dp.profit > min_profit,
				dps
			))
		if max_profit is not None:
			dps = list(filter(
				lambda dp: dp.profit < max_profit,
				dps
			))

		if min_duration is not None:
			dps = list(filter(
				lambda dp: dp.duration >= min_duration,
				dps
			))

		return dps

	def test_plot_profit_vs_loss(self):
		dps = sorted(self.__filter_stats(
				self.__get_valid_dps(),
				min_profit=-5,
				max_profit=5
				# time=datetime.now() - timedelta(hours=33),
				# model_losses=(1.5, None, None)
			),
			key=lambda dp: dp.profit,
			reverse=True
		)
		print(f"Using {len(dps)} dps")
		self.__print_dps(dps)
		self.assertGreater(len(dps), 0)
		losses = [
			[dp.model_losses[i] for dp in dps]
			for i in range(len(dps[0].model_losses))
		]
		losses.append(
			[np.prod(dp.model_losses) for dp in dps]
		)

		names = [
			"nn.CrossEntropyLoss()",
			"ProximalMaskedLoss",
			"MeanSquaredClassError",
			"ReverseMAWeightLoss(window_size=10, softmax=True)",
			"PredictionConfidenceScore(softmax=True)",
			"OutputClassesVariance(softmax=True)",
			"OutputBatchVariance(softmax=True)",
			"OutputBatchClassVariance(softmax=True)",
			"Product of all losses"
		]
		for i in range(len(losses)):
			print(f"Plotting {names[i]}")
			plt.figure()
			plt.title(names[i])
			plt.scatter(
				losses[i],
				[dp.profit for dp in dps],
			)

			plt.axhline(y=0, color="black")
		plt.show()

	def test_plot_losses(self):
		dps = self.repository.retrieve_by_loss_complete()
		print(f"Using {len(dps)} dps")
		plt.scatter(
			*[
				[dp.model_losses[i] for dp in dps]
				for i in [0, -1]
			]
		)
		plt.show()

	def test_get_all(self):
		stats = self.repository.retrieve_all()
		self.__print_dps(stats)
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

	def test_clear_losses(self):
		stats = self.repository.retrieve_all()
		for i, stat in enumerate(stats):
			stat.model_losses = [
				0
				if i in [7] else stat.model_losses[i]
				for i in range(len(stat.model_losses))
			]
			self.repository.store(stat)
			print(f"Progress: {(i + 1) * 100 / len(stats):.2f}%")

	def test_add_empty_loss(self):
		stats = self.repository.retrieve_all()
		for i, stat in enumerate(stats):
			if len(stat.model_losses) == 7:
				stat.model_losses += (0.0,)
				self.repository.store(stat)
			print(f"Progress: {(i + 1) * 100 / len(stats):.2f}%")

	def test_single_allocate(self):
		stat = self.repository.allocate_for_runlive(
			allow_locked=True
		)
		print(f"Allocated {stat.id}")
		self.repository.finish_session(stat, 0)
		self.__print_dps([stat])

	def test_multiple_allocate(self):

		allocated = []
		for i in range(5):
			allocated.append(self.repository.allocate_for_runlive())
			self.repository.finish_session(allocated[-1], 0)
			print(f"Allocated {len(allocated)}")

		self.__print_dps(allocated)

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
		dps = sorted(
			self.__filter_stats(
				self.__get_valid_dps(),
				# time=datetime.now() - timedelta(hours=),
				model_losses=(4.5,),
				max_profit=0
			),
			key=lambda dp: dp.model_losses[0]
		)

		self.__print_dps(dps)

	def test_get_sessions(self):
		dps = sorted(
			self.__filter_stats(
				self.repository.retrieve_all(),
				model_key='linear',
				# model_losses=(1.5,None),
				# time=datetime.now() - timedelta(hours=9),
			),
			key=lambda dp: datetime.now() - timedelta(days=1000) if len(dp.session_timestamps) == 0 else dp.session_timestamps[-1],
			reverse=True
		)

		self.__print_dps(dps)

	def test_get_custom_sorted(self):
		dps = sorted(
			self.__filter_stats(
				self.repository.retrieve_valid(),
				model_key='linear',
				# model_losses=(1.5,None),
				# time=datetime.now() - timedelta(hours=9),
			),
			key=lambda dp: dp.profit,
			reverse=True
		)

		self.__print_dps(dps)

	def test_retrieve_non_locked(self):
		dps = self.repository.retrieve_all()
		non_locked = self.repository.retrieve_non_locked()
		locked = [
			stat
			for stat in dps
			if stat.id not in [stat.id for stat in non_locked]
		]
		print("All:")
		self.__print_dps(dps)

		print("Non Locked:")
		self.__print_dps(non_locked)

		print("Locked:")
		self.__print_dps(locked)

	def test_reset_sessions(self):

		stats = self.repository.retrieve_all()
		for i, stat in enumerate(stats):
			stat.profits = []
			stat.session_timestamps = []
			stat.duration = 0
			self.repository.store(stat)
			print(f"Progress: {(i + 1) * 100 / len(stats):.2f}%")

	def test_plot_distribution(self):

		def count_bounds(values: typing.List[float], bounds: typing.List[typing.Tuple[float, float]]):
			bound_counts = [0 for _ in bounds]

			for value in values:
				for i, (b_b, b_t) in enumerate(bounds):
					if b_b <= value <= b_t:
						bound_counts[i] += 1
			return bound_counts

		def generate_bounds(values: typing.List[float], size: int):

			sequence = np.linspace(
				min(values),
				max(values),
				size+1
			)

			return [
				(sequence[i], sequence[i+1])
				for i in range(size)
			]

		def process_loss(losses: typing.List[float], name: str):

			bounds = generate_bounds(losses, 10)
			counts = count_bounds(losses, bounds)

			plt.figure()
			plt.title(name)
			plt.scatter(
				[sum(bound)/2 for bound in bounds],
				counts
			)

		dps = list(filter(
			lambda dp: (
					len(dp.model_losses) == len(self.loss_names)
			),
			self.repository.retrieve_valid()
		))

		print(f"Using {len(dps)} stats")

		for i in range(len(self.loss_names)):
			losses = [dp.model_losses[i] for dp in dps]
			process_loss(losses, self.loss_names[i])

		plt.show()

	def test_trim_stats(self):

		BOUNDS = (0, 15)
		LOSS_IDX = 0

		all = self.repository.retrieve_all()
		for i, dp in enumerate(all):
			if not (BOUNDS[0] <= dp.model_losses[LOSS_IDX] <= BOUNDS[1]):
				print(f"Deleting {dp.model_losses[LOSS_IDX]}")
				self.repository.delete(dp.id)
			print(f"{(i+1)*100/len(all) :.2f}%... Done")

	def test_get_selected(self):
		stats = self.repository.retrieve_valid()
		self.__print_dps(stats)

	def test_wipe_profits(self):

		IDS = [
		]

		stats = list(filter(
			lambda stat: stat.id in IDS,
			self.repository.retrieve_all()
		))
		print(f"Wiping {len(stats)} stats")

		for i, stat in enumerate(stats):
			print(f"Wiping {stat.id}")
			stat.profits = []
			stat.session_timestamps = []
			stat.duration = 0
			self.repository.store(stat)
			print(f"Progress: {(i + 1) * 100 / len(stats):.2f}%")

	def test_wipe_old_profits(self):
		date_threshold = datetime(
			year=2024,
			month=10,
			day=16,
		)

		stats = self.__filter_stats(
			self.repository.retrieve_all(),
			min_duration=1
		)

		for i, stat in enumerate(stats):
			valid_idxs = [
				i
				for i, time in enumerate(stat.session_timestamps)
				if time >= date_threshold
			]
			if len(valid_idxs) == len(stat.session_timestamps):
				continue
			stat.profits, stat.session_timestamps = [
				[
					array[i]
					for i in valid_idxs
					if i < len(array)
				]
				for array in (stat.profits, stat.session_timestamps)
			]
			self.repository.store(stat)
			print(f"Progress: {(i + 1) * 100 / len(stats):.2f}%")

	def test_transfer_profits(self):

		class TargetNotFoundException(Exception):
			pass

		def get_target(stat: RunnerStats) -> RunnerStats:
			target = list(filter(
				lambda s: s.model_name == stat.model_name and s.temperature == 1.0,
				all
			))
			if len(target) == 0:
				raise TargetNotFoundException()
			return target[0]

		def is_transferable(stat: RunnerStats) -> bool:
			return stat.temperature != 1 and len(stat.session_timestamps) > 0

		def get_transferable() -> typing.List[RunnerStats]:
			return list(filter(is_transferable, all))

		def transfer(source: RunnerStats, target: RunnerStats):
			print(f"[+]Transferring {source.id} to {target.id}")

			for target_arr, source_arr in zip([target.session_timestamps, target.profits], [source.session_timestamps, source.profits]):
				target_arr.extend(source_arr)

			source.session_timestamps, source.profits = [], []
			self.repository.store(target)
			self.repository.store(source)

		def process(stat):
			try:
				target = get_target(stat)
				transfer(stat, target)
			except TargetNotFoundException:
				print(f"[-]Could not transfer {stat.id}")

		all = self.repository.retrieve_all()

		transferable = get_transferable()

		for i, stat in enumerate(transferable):
			process(stat)
			print(f"{(i+1)*100/len(transferable):.2f}%... Done")

	def test_sync_duration(self):

		SESSION_LEN = 3*60*60

		stats = self.repository.retrieve_all()

		for i, stat in enumerate(stats):
			stat.duration = SESSION_LEN * len(stat.session_timestamps)
			self.repository.store(stat)

			print(f"{i+1} of {len(stats)}... Done")
