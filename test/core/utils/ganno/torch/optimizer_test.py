import unittest

from torch.utils.data import DataLoader

from core import Config
from core.utils.ganno.torch.optimizer import Optimizer, CNNOptimizer, TransformerOptimizer
from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.training.callbacks.checkpoint_callback import StoreCheckpointCallback
from lib.utils.file_storage import PCloudClient


class OptimizerTest(unittest.TestCase):

	def test_cnn_optimizer(self):
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

		trainer_callbacks = [
			StoreCheckpointCallback(
				path="/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/",
				fs=PCloudClient(Config.PCLOUD_API_TOKEN, "/Apps/RTrader/Models")
			)
		]

		optimizer = CNNOptimizer(
			vocab_size=449,
			dataset=dataset,
			test_dataset=test_dataset,
			epochs=2,
			population_size=5,
			trainer_callbacks=trainer_callbacks
		)

		optimizer.start(epochs=10)

	def test_transformer_optimizer(self):
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
		optimizer = TransformerOptimizer(
			vocab_size=449,
			dataset=dataset,
			test_dataset=test_dataset,
			epochs=2,
			population_size=5
		)

		optimizer.start(epochs=10)

