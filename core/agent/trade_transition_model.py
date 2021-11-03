from typing import *

import numpy as np
import tensorflow as tf
from tensorflow import keras


class TransitionModel(keras.Model):

	def __init__(self):
		super(TransitionModel, self).__init__()
		self.input_layer = keras.layers.Input(shape=64)
		self.hidden_layer = keras.layers.Dense(100)
		self.output_layer = keras.layers.Dense(1)

	def call(self, inputs):
		return self.output_layer(
			self.hidden_layer(
				self.input_layer(inputs)
			)
		)
