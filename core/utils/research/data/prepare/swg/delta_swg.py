import typing

import numpy as np

import os

from lib.utils.logger import Logger
from .abstract_swg import AbstractSampleWeightGenerator


class DeltaSampleWeightGenerator(AbstractSampleWeightGenerator):

	def __init__(
			self,
			*args,
			bounds: typing.Union[typing.List[float], np.ndarray],
			alpha: float = 7,
			X_extra_len: int = 124,
			y_extra_len: int = 1,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__alpha = alpha
		self.__X_extra_len = X_extra_len
		self.__y_extra_len = y_extra_len

		if isinstance(bounds, list):
			bounds = np.array(bounds)
		self.__bounds = bounds

	def _generate_weights(self, X: np.ndarray, y: np.ndarray):
		X, y = [arr[:, :-extra_len] for arr, extra_len in zip([X, y], [self.__X_extra_len, self.__y_extra_len])]
		y = self.__bounds[np.argmax(y, axis=1)]
		return 10**(((((X[:, -1]/X[:, -2]) - 1) * (y - 1))*-1) * 10 ** self.__alpha)
