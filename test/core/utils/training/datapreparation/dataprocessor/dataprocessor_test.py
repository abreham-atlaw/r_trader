import typing

import numpy as np
from tensorflow import keras

import unittest
from unittest.mock import MagicMock

from core.utils.training.datapreparation.dataprocessor import DataProcessor, GranularDataProcessor


class Model:

	def __init__(self, l):
		self.l = l
		self.input_shape = (None, 5)

	def predict(self, sequence):
		out = np.zeros((sequence.shape[0], self.l))
		for i in range(sequence.shape[0]):
			out[i, 2] = 1
		return out


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

	def test_gran(self):
		depth = False
		bounds = [i for i in range(10)]
		generator = MagicMock()
		generator.__getitem__ = MagicMock(return_value=np.sin(np.arange(1100)).reshape((10, 110)))
		generator.__len__ = MagicMock(return_value=30)
		model = Model(len(bounds))
		processor = GranularDataProcessor(
			generator,
			model,
			bounds,
			32,
			32,
			depth_input=depth
		)
		for i in range(5):
			element = processor.get_data(0, i)
			x, y = element[0]
			print(element)
