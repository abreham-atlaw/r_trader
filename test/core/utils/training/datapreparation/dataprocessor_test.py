import typing

import numpy as np
from tensorflow import keras

import unittest
from unittest.mock import MagicMock

from core.utils.training.datapreparation.dataprocessor import DataProcessor


class DataProcessorTest(unittest.TestCase):

	@staticmethod
	def __generate_model(delta=False, depth=True):
		model = keras.Sequential()
		model.add(keras.layers.InputLayer(100 + int(depth) + int(delta)))
		model.add(keras.layers.Dense(10))
		model.add(keras.layers.Dense(1))
		model.compile(optimizer="adam", loss="mse")
		return model

	def test_functionality(self):
		depth = False
		generator = MagicMock()
		generator.__getitem__ = MagicMock(return_value=np.sin(np.arange(1100)).reshape((10, 110)))
		generator.__len__ = MagicMock(return_value=30)
		core_model, delta_model = self.__generate_model(depth=depth), self.__generate_model(depth=depth, delta=True)
		processor = DataProcessor(
			generator,
			core_model,
			delta_model,
			32,
			32,
			depth_input=depth
		)
		for i in range(5):
			element = processor.get_data(0, i)
			self.assertTrue(isinstance(element, tuple))

