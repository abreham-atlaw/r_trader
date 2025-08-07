from abc import ABC, abstractmethod

import numpy as np


class SmoothingAlgorithm(ABC):

	_INITIAL_SIZE = 1024

	@abstractmethod
	def apply(self, x: np.ndarray) -> np.ndarray:
		pass

	@property
	def reduction(self) -> int:
		initial_size = self._INITIAL_SIZE
		out_size = self.apply(np.zeros(initial_size)).shape[0]
		return initial_size - out_size

	def apply_on_batch(self, x: np.ndarray) -> np.ndarray:
		return np.stack([
			self.apply(x[i])
			for i in range(x.shape[0])
		])

	def __call__(self, x: np.ndarray) -> np.ndarray:
		return self.apply(x)

	def __str__(self):
		return self.__class__.__name__
