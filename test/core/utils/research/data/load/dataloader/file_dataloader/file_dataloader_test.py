import time
import unittest

from core.utils.research.data.load.dataloader.file_dataloader import FileDataLoader
from core.utils.research.data.load.flm.file_loader import FileLoader
from lib.utils.devtools import performance
from lib.utils.logger import Logger


class FileDataLoaderTest(unittest.TestCase):

	def setUp(self):
		self.root_dirs = ["/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/dummy_64"]

		self.loader = FileDataLoader(
			file_loader=FileLoader(
				root_dirs=self.root_dirs,
				use_pool=False
			)
		)

	def tearDown(self):
		pass

	def __load_data(self, loader):
		for X, y in loader:
			pass

	def test_functionality(self):
		performance.track_performance(
			"self.__load_data(self.loader)",
			lambda: self.__load_data(self.loader)
		)
		Logger.info(performance.durations)

	def test_optimal_num_workers(self):
		for i in range(0, 5):
			print(f"Evaluating Workers = {i}")
			loader = FileDataLoader(
				root_dirs=self.root_dirs,
				pool_size=1000,
				workers=i,
				preload_size=30
			)
			key = f"Num Workers: {i}"
			performance.track_performance(
				key,
				lambda: self.__load_data(loader)
			)
			print(f"Performance: {performance.durations[key]}")

		print(performance.durations)
