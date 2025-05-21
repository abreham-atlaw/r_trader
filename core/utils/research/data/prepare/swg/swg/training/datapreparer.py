import typing

import numpy as np


class SampleWeightGeneratorDataPreparer:

	def __init__(
			self,
			bounds: typing.List[float],
			X_extra_len: int = 124,
			y_extra_len: int = 1,
	):
		self.__X_extra_len = X_extra_len
		self.__y_extra_len = y_extra_len

		if isinstance(bounds, list):
			bounds = np.array(bounds)
		self.__bounds = bounds

	def prepare(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		X, y = [X[:, :-self.__X_extra_len], y[:, :-self.__y_extra_len]]
		y = X[:, -1] * self.__bounds[np.argmax(y, axis=1)]
		return np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
