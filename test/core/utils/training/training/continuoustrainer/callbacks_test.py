import typing

from tensorflow import keras

import unittest
from unittest.mock import MagicMock

from core.utils.training.training.continuoustrainer.callbacks import PCloudContinuousTrainerCheckpointCallback
from core.utils.training.training.trainer import Trainer
from core import Config


class PCloudContinuousTrainerCheckpointCallbackTest(unittest.TestCase):

	__ID = "34"
	__URLS = ("https://example.com/url0", "https://example.com/url1")
	__EPOCH = 4
	__STATE = Trainer.State(__EPOCH, 0, 0, 0)

	def setUp(self) -> None:
		self.__repository = MagicMock()
		self.__repository.update_checkpoint = MagicMock(return_value=None)
		self.__repository.get_checkpoint = MagicMock(return_value=(self.__URLS, self.__EPOCH))

	def __generate_mock_model(self) -> keras.Model:
		model = keras.Sequential()
		model.add(keras.layers.InputLayer(100))
		model.add(keras.layers.Dense(10))
		model.add(keras.layers.Dense(1))
		model.compile(optimizer="adam", loss="mse")
		return model

	def test_functionality(self):
		callback = PCloudContinuousTrainerCheckpointCallback(base_path="/Apps/RTrader")
		callback.init(self.__ID, self.__repository)
		callback.on_epoch_end(
			self.__generate_mock_model(),
			self.__generate_mock_model(),
			self.__STATE
		)

		self.__repository.update_checkpoint.assert_called_with(self.__ID, unittest.mock.ANY, self.__EPOCH)

