from typing import *

from tensorflow import keras


class TransitionModel(keras.Model):

	def __init__(self):
		super(TransitionModel, self).__init__()
		self.input_layer = keras.layers.InputLayer(input_shape=64)
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
