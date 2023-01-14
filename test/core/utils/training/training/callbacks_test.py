import typing
from abc import ABC, abstractmethod

from tensorflow import keras

import unittest

from lib.utils.file_storage import PCloudClient
from core.utils.training.training.trainer import Trainer
from core.utils.training.training.callbacks import PCloudCheckpointUploadCallback
from core import Config


class PCloudCheckpointUploadCallbackTest(unittest.TestCase):

	def setUp(self) -> None:
		self.__path = "/Apps"
		self.pCloudClient = PCloudClient(Config.PCLOUD_API_TOKEN, self.__path)

	def __generate_mock_model(self) -> keras.Model:
		model = keras.Sequential()
		model.add(keras.layers.InputLayer(100))
		model.add(keras.layers.Dense(10))
		model.add(keras.layers.Dense(1))
		model.compile(optimizer="adam", loss="mse")
		return model

	def __generate_mock_state(self) -> Trainer.State:
		return Trainer.State(0, 0, 0, 0)

	def test_functionality(self):
		callback = PCloudCheckpointUploadCallback(self.__path)
		callback.on_epoch_end(self.__generate_mock_model(), self.__generate_mock_model(), self.__generate_mock_state())

