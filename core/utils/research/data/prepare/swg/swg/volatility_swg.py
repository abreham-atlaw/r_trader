import typing

import numpy as np

from .basic_swg import BasicSampleWeightGenerator


class VolatilitySampleWeightGenerator(BasicSampleWeightGenerator):

	def __init__(
			self,
			*args,
			bounds: typing.Union[typing.List[float], np.ndarray],
			p=48,
			X_extra_len: int = 124,
			y_extra_len: int = 1,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__p = p
		self.__X_extra_len = X_extra_len
		self.__y_extra_len = y_extra_len
		if isinstance(bounds, list):
			bounds = np.array(bounds)
		self.__bounds = bounds

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		X, y = [arr[:, :-extra_len] for arr, extra_len in zip([X, y], [self.__X_extra_len, self.__y_extra_len])]
		y = X[:, -1] * self.__bounds[np.argmax(y, axis=1)]
		return (10**(np.abs((y - X[:, -1]) - (X[:, -1] - X[:, -2]))**(1/2)))**self.__p
