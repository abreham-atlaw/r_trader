import numpy as np

from .layer import Layer


class MinMaxNorm(Layer):

	def __init__(self, axis=-1):
		self.__axis = axis

	def _call(self, x: np.ndarray) -> np.ndarray:
		x = x - np.min(x, axis=self.__axis, keepdims=True)
		x = x / np.max(x, axis=self.__axis, keepdims=True)
		return x
