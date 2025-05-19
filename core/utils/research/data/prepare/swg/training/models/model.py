import pickle
from abc import ABC, abstractmethod

import numpy as np

from core.utils.research.data.prepare.swg.training.models.layers import MinMaxNorm, Identity


class SampleWeightGenerationModel(ABC):

	def __init__(
			self,
			norm: bool = False
		):
		self.norm = MinMaxNorm() if norm else Identity()

	def _process_input(self, X: np.ndarray) -> np.ndarray:
		return self.norm(X)

	@abstractmethod
	def _predict(self, X: np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def _fit(self, X: np.ndarray, y: np.ndarray):
		pass

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self._predict(self._process_input(X))

	def fit(self, X: np.ndarray, y: np.ndarray):
		self._fit(self._process_input(X), y)

	def save(self, path: str):
		with open(path, 'wb') as file:
			pickle.dump(self, file)

	@classmethod
	def load(cls, path: str):
		with open(path, 'rb') as file:
			return pickle.load(file)

	def __call__(self, X: np.ndarray) -> np.ndarray:
		return self.predict(X)
