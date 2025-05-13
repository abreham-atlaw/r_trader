import pickle
from abc import ABC, abstractmethod

import numpy as np


class SampleWeightGenerationModel(ABC):

	@abstractmethod
	def predict(self, X: np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	def save(self, path: str):
		with open(path, 'wb') as file:
			pickle.dump(self, file)

	@classmethod
	def load(cls, path: str):
		with open(path, 'rb') as file:
			return pickle.load(file)

	def __call__(self, X: np.ndarray) -> np.ndarray:
		return self.predict(X)
