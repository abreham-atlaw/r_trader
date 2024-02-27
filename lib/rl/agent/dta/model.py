from abc import ABC, abstractmethod

import numpy as np
from tensorflow.python import keras
import torch
import torch.nn as nn

from lib.utils.torch_utils.model_handler import ModelHandler


class Model(ABC):

	@abstractmethod
	def predict(self, inputs: np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	@abstractmethod
	def save(self, path: str):
		pass

	@classmethod
	@abstractmethod
	def load(cls, path: str) -> 'Model':
		pass

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)


class KerasModel(Model):

	def __init__(self, model: keras.Model):
		self.__model = model

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.__model.predict(inputs)

	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	def save(self, path: str):
		pass

	@classmethod
	def load(cls, path: str) -> 'Model':
		pass


class TorchModel(Model):

	def __init__(self, model: nn.Module):
		self.__model = model

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.__model(torch.from_numpy(inputs.astype(np.float32))).detach().numpy()

	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	def save(self, path: str):
		pass

	@classmethod
	def load(cls, path: str) -> 'Model':
		return TorchModel(ModelHandler.load(path))

