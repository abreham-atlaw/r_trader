import time
import unittest

from core.utils.research.data.load.dataloader.file_dataloader import FileDataLoader
from core.utils.research.data.load.dataloader.file_dataloader.cfl import ConcurrentFileDataLoader
from core.utils.research.data.load.flm.file_loader import FileLoader
from lib.utils.devtools import performance
from lib.utils.logger import Logger


class ConcurrentFileDataLoaderTest(unittest.TestCase):

	def setUp(self):
		self.root_dirs = ["/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/dummy_64"]

		self.loader = ConcurrentFileDataLoader(
			root_dirs=self.root_dirs,
			workers=1,
		)

	def tearDown(self):
		pass

	def __load_data(self, loader):
		i = 0
		for X, y in loader:
			i += 1
		self.assertEqual(len(loader), i)

	def test_functionality(self):
		performance.track_performance(
			"self.__load_data(self.loader)",
			lambda: self.__load_data(self.loader)
		)
		Logger.info(performance.durations)

	def test_optimal_num_workers(self):
		for i in range(1, 6):
			print(f"Evaluating Workers = {i}")
			loader = ConcurrentFileDataLoader(
				root_dirs=self.root_dirs,
				workers=i,
				prefetch_factor=100
			)
			key = f"Num Workers: {i}"
			performance.track_performance(
				key,
				lambda: self.__load_data(loader)
			)
			print(f"Performance: {performance.durations[key]}")

		print(performance.durations)
