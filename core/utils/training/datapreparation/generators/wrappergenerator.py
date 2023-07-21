from typing import *

import numpy as np
from tensorflow.keras.utils import Sequence as KerasSequence

import gc


class WrapperGenerator(KerasSequence):

	def __init__(self, batch_size):
		self.__batch_size = batch_size
		self.__slice_point = None
		self.__data = None
		self.__y_reshape = False

	def __init_data(self, data: Tuple[np.ndarray, np.ndarray]):
		self.__slice_point = data[0].shape[1]
		if len(data[0].shape) != len(data[1].shape):
			self.__y_reshape = True
		self.__data = self.__concat_values(data)

	def add_data(self, data: Tuple[np.ndarray, np.ndarray]):
		if self.__data is None:
			self.__init_data(data)
			return
		self.__data = np.concatenate(
			(
				self.__data,
				self.__concat_values(data)
			)
		)

	def merge(self, generator: 'WrapperGenerator'):
		for i in range(len(generator)):
			self.add_data(generator[i])

	def __concat_values(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
		if self.__y_reshape:
			data = data[0], data[1].reshape((-1, 1))
		return np.concatenate(data, axis=1)

	def __separate_values(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		X, y = data[:, :self.__slice_point], data[:, self.__slice_point:]
		if self.__y_reshape:
			y = y.reshape((-1,))
		return X, y

	def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
		return self.__separate_values(
			self.__data[idx * self.__batch_size: (idx + 1) * self.__batch_size]
		)

	def __len__(self):
		return int(np.ceil(len(self.__data) / self.__batch_size))

	def shuffle(self):
		np.random.shuffle(self.__data)

	def destroy(self):
		del self.__data
		gc.collect()
