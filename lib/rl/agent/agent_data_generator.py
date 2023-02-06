import typing

import numpy as np
from tensorflow.keras.utils import Sequence as KerasSequence

import math


class AgentDataGenerator(KerasSequence):

	def __init__(self, batch_size: int, shape: typing.Tuple[typing.Tuple, typing.Tuple] = None):
		self.__batch_size = batch_size
		self.__X, self.__y = None, None
		if shape is not None:
			self.set_shape(shape)

	def __get_idxs(self, idx: int) -> typing.Tuple[int, int]:
		return idx*self.__batch_size, (idx+1)*self.__batch_size

	def set_shape(self, shape: typing.Tuple[typing.Tuple, typing.Tuple]):
		self.__X, self.__y = np.zeros((0, *shape[0])), np.zeros((0, *shape[1]))

	def concatenate(self, X: np.ndarray, y: np.ndarray):
		if self.__X is None:
			self.set_shape((X.shape[1:], y.shape[1:]))
		self.__X = np.concatenate((self.__X, X))
		self.__y = np.concatenate((self.__y, y))

	def append(self, X: np.ndarray, y: np.ndarray):
		self.concatenate(np.expand_dims(X, 0), np.expand_dims(y, 0))

	def remove(self, idx):
		si, ei = self.__get_idxs(idx)
		self.__X = np.concatenate((self.__X[:si], self.__X[ei:]))
		self.__y = np.concatenate((self.__y[:si], self.__y[ei:]))

	def __getitem__(self, idx: int):
		si, ei = self.__get_idxs(idx)
		return self.__X[si: ei], self.__y[si: ei]

	def __len__(self):
		return math.ceil(self.__X.shape[0] / self.__batch_size)

