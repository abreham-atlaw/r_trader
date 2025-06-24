import typing
from abc import ABC, abstractmethod

import numpy as np


class AbstractSampleWeightGenerator(ABC):

	def __init__(
			self,
			min_weights: float = 0.0
	):
		self.__min_weights = min_weights

	@abstractmethod
	def _generate(self, *args, **kwargs) -> np.ndarray:
		pass

	def generate(self, *args, **kwargs) -> np.ndarray:
		weights = self._generate(*args, **kwargs)
		weights[weights < self.__min_weights] = self.__min_weights
		return weights

	def __call__(self, *args, **kwargs):
		return self.generate(*args, **kwargs)