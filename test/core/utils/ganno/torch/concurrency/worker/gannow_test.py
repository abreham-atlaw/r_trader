import unittest

from core import Config
from core.utils.ganno.torch.concurrency.queen.gannoq import CNNOptimizerQueen
from core.utils.ganno.torch.concurrency.worker.gannow import CNNOptimizerWorker
from core.utils.research.data.load.dataset import BaseDataset


class GannoWorkerTest(unittest.TestCase):

	def test_functionality(self):
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		test_dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)

		worker = CNNOptimizerWorker(
			Config.GANNO_SERVER_URL,
			vocab_size=449,
			dataset=dataset,
			test_dataset=test_dataset,
			epochs=2,
			population_size=5,
			train_timeout=10
		)

		worker.start(
			epochs=10
		)
