from typing import *
from unittest.mock import MagicMock

import numpy as np
from tensorflow import keras

import unittest

from core.utils.training.datapreparation import DataProcessor
from lib.network.oanda import Trader
from core.utils.ganno import NNGeneticAlgorithm, GannoTrainer
from core.utils.ganno.nnconfig import ModelConfig, ConvPoolLayer
from core import Config


class TestGannoTrainer(GannoTrainer):

	def _init_processor(self, core_model, delta_model) -> DataProcessor:
		generator = MagicMock()
		generator.__getitem__ = MagicMock(return_value=np.sin(np.arange(1100)).reshape((10, 110)))
		generator.__len__ = MagicMock(return_value=30)
		return DataProcessor(generator, core_model, delta_model, 32, 32)


class NNGeneticAlgorithmTest(unittest.TestCase):

	def setUp(self) -> None:
		trader = Trader(Config.OANDA_TOKEN, Config.OANDA_TEST_ACCOUNT_ID)
		trainer = TestGannoTrainer(3, epochs=2)
		self.optimizer = NNGeneticAlgorithm(trainer)

	def test_validate(self):
		config = ModelConfig(
			100,
			[10, 10, 10],
			[ConvPoolLayer(10, 5, 10), ConvPoolLayer(10, 0, 0)],
			True,
			True,
			"relu",
			"relu",
			"mse",
			"adam"
		)
		self.assertFalse(config.validate())

	def test_functionality(self):
		self.optimizer.start(5)


if __name__ == "__main__":
	unittest.main()
