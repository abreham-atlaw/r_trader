import unittest

import torch

from core.di import ServiceProvider
from core.utils.research.training.trackers.stats_tracker import DynamicStatsTracker, Keys


class DynamicStatsTrackerTest(unittest.TestCase):

	def setUp(self):
		self.dump_size = 5
		self.trackers = {
			key: DynamicStatsTracker(
				key,
				model_name="test-model",
				fs=ServiceProvider.provide_file_storage("/Apps/RTrader/stats/"),
				dump_size=self.dump_size,
				dump_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/training_stat_dumps"
			) for key in Keys.ALL
		}

	def test_dump_X(self):

		for b in range(self.dump_size + 2):
			self.trackers.get(
				Keys.X
			).on_batch_end(
				*([torch.rand((10, 100)) for _ in range(len(Keys.ALL))]),
				batch=b,
				epoch=1
			)

	def test_dump_all(self):

		for b in range(self.dump_size + 2):
			for key in Keys.ALL:
				self.trackers.get(
					key
				).on_batch_end(
					*([torch.rand((10, 100)) for _ in range(len(Keys.ALL))]),
					batch=b,
					epoch=1
				)
