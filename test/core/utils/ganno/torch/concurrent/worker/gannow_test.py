import unittest

from core.utils.ganno.torch.concurrent.queen.gannoq import CNNOptimizerQueen
from core.utils.ganno.torch.concurrent.worker.gannow import CNNOptimizerWorker
from core.utils.research.data.load.dataset import BaseDataset


class GannoWorkerTest(unittest.TestCase):

	URL = "https://ga-server.vercel.app/"

	def test_functionality(self):
		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train"
			],
		)
		test_dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/test"
			],
		)

		worker = CNNOptimizerWorker(
			self.URL,
			vocab_size=449,
			dataset=dataset,
			test_dataset=test_dataset,
			epochs=2,
			population_size=5
		)

		worker.start(
			epochs=10
		)
