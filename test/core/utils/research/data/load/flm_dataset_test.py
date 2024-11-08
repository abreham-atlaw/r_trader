import unittest
from datetime import datetime

from torch.utils.data import DataLoader

from core.utils.research.data.load import FLMDataset, BaseDataset
from core.utils.research.data.load.flm import FileLoadManager
from core.utils.research.data.load.flm.file_loader import FileLoader


class FLMDatasetTest(unittest.TestCase):

	def setUp(self):

		self.root_dirs = ["/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/dummy"]

		self.manager = FileLoadManager(
			FileLoader(
				root_dirs=self.root_dirs,
				pool_size=3
			)
		)

		self.dataset = FLMDataset(
			self.manager
		)

		self.other_dataset = BaseDataset(
			root_dirs=self.root_dirs
		)

	def test_functionality(self):
		X, y = self.dataset[1500]
		self.assertEqual(X.shape[0], 1024)
		self.assertEqual(y.shape[0], 432)

	def test_dataloader(self):
		self.__benchmark_dataset(self.dataset)

	def tearDown(self):
		self.manager.stop()

	def __benchmark_dataset(self, dataset):

		print(f"Benchmarking {dataset.__class__.__name__}")

		start_time = datetime.now()

		dataloader = DataLoader(
			dataset=dataset,
			batch_size=64,
			num_workers=10
		)

		for X, y in dataloader:
			self.assertIsNotNone(X)

		print(f"Duration {(datetime.now() - start_time).total_seconds()}")

	def test_compare_dataset_performances(self):

		for dataset in [self.other_dataset, self.dataset]:
			self.__benchmark_dataset(dataset)