from typing import *

from tensorflow import keras
import numpy as np

from lib.dnn.layers import Delta, Percentage, MovingAverage, Norm
from core import Config


class TransitionModel(keras.Model):

	def __init__(self):
		super(TransitionModel, self).__init__()
		self.input_layer = keras.layers.InputLayer(input_shape=Config.MARKET_STATE_MEMORY)
		self.hidden_layers = [
			keras.layers.Dense(128),
			keras.layers.Dense(1024),
			keras.layers.Dense(64)
		]
		self.output_layer = keras.layers.Dense(1, activation="sigmoid")

	def call(self, inputs):

		output = self.input_layer(inputs)
		for layer in self.hidden_layers:
			output = layer(output)
		return self.output_layer(output)

	@staticmethod
	def load_model(path: str) -> keras.Model:
		return keras.models.load_model(path, custom_objects={layer.__name__: layer for layer in [Delta, MovingAverage, Percentage, Norm]})


class RemoteTransitionModel:

	def __init__(self, address: str = Config.REMOTE_TRADER_URL):
		self.__address = address

	def predict(self, X: np.ndarray):
		pass
