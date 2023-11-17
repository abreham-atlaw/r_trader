from typing import *

from tensorflow import keras

import unittest

from lib.network.oanda import Trader
from core.utils.ganno import NNGeneticAlgorithm, LiveTrainer
from core.utils.ganno.nnconfig import ModelConfig, ConvPoolLayer
from core import Config


class NNGeneticAlgorithmTest(unittest.TestCase):

	def setUp(self) -> None:
		trader = Trader(Config.OANDA_TOKEN, Config.OANDA_TEST_ACCOUNT_ID)
		trainer = LiveTrainer(trader, 500, instruments=[("AUD", "USD"), ("EUR", "USD"), ("USD", "CAD")], epochs=2, average_window=10)
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
