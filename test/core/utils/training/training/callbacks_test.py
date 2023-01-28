import typing
from abc import ABC, abstractmethod

from tensorflow import keras

import unittest
import math

from lib.utils.file_storage import PCloudClient
from core.utils.training.training.trainer import Trainer
from core.utils.training.training.callbacks import PCloudCheckpointUploadCallback, EarlyStoppingCallback
from core import Config

import matplotlib.pyplot as plt


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


class EarlyStoppingCallbackTest(unittest.TestCase):


	def test_min_loss_pass(self):
		callback = EarlyStoppingCallback(model=0, patience=8, source=1, mode=EarlyStoppingCallback.Modes.MIN)
		metrics = Trainer.MetricsContainer()
		for i in range(20):
			metrics.add_metric(Trainer.Metric(
				source=1,
				model=0,
				epoch=i,
				depth=5,
				value=(math.sin((i*3.14*2/20)+(3.14*1.5/2)),)
			))

		early_stopped = False
		try:
			callback.on_epoch_end(None, None, Trainer.State(0, 0, 0, 5), metrics)
		except EarlyStoppingCallback.EarlyStopException:
			early_stopped = True

		self.assertFalse(early_stopped)

	def test_min_loss_stop(self):
		callback = EarlyStoppingCallback(model=0, patience=8, source=1, mode=EarlyStoppingCallback.Modes.MIN)
		metrics = Trainer.MetricsContainer()
		for i in range(20):
			metrics.add_metric(Trainer.Metric(
				source=1,
				model=0,
				epoch=i,
				depth=5,
				value=(math.sin((i*3.14*2/20)+(3.14/2)),)
			))

		early_stopped = False
		try:
			callback.on_epoch_end(None, None, Trainer.State(0, 0, 0, 5), metrics)
		except EarlyStoppingCallback.EarlyStopException:
			early_stopped = True

		self.assertTrue(early_stopped)