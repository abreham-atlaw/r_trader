import typing

import numpy as np

from .abstract_swg import AbstractSampleWeightGenerator


class LambdaSampleWeightGenerator(AbstractSampleWeightGenerator):

	def __init__(
			self,
			*args,
			func: typing.Callable,
			bounds: typing.Union[typing.List[float], np.ndarray],
			X_extra_len: int = 124,
			y_extra_len: int = 1,
			pass_y_value: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__func = func
		self.__X_extra_len = X_extra_len
		self.__y_extra_len = y_extra_len
		if isinstance(bounds, list):
			bounds = np.array(bounds)
		self.__bounds = bounds
		self.__pass_y_value = pass_y_value

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		X, y = [arr[:, :-extra_len] for arr, extra_len in zip([X, y], [self.__X_extra_len, self.__y_extra_len])]
		if self.__pass_y_value:
			y = X[:, -1] * self.__bounds[np.argmax(y, axis=1)]
		return self.__func(X, y)
