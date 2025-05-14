from abc import ABC, abstractmethod

import numpy as np


class SmoothingAlgorithm(ABC):

	@abstractmethod
	def apply(self, x: np.ndarray) -> np.ndarray:
		pass

	def __call__(self, x: np.ndarray) -> np.ndarray:
		return self.apply(x)

	def __str__(self):
		return self.__class__.__name__
