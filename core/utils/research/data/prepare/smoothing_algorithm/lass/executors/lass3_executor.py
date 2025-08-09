import typing

import numpy as np

from core.utils.research.model.model.savable import SpinozaModule
from .lass_executor import LassExecutor


class Lass3Executor(LassExecutor):

	def __init__(
			self,
			*args,
			padding: int = 0,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__padding = padding
		self.__target_size = None

	def set_model(self, model: SpinozaModule):
		super().set_model(model)
		self.__target_size = self._window_size - 2 * self.__padding

	def __get_source_block(self, x: np.ndarray, target: typing.Tuple[int, int]) -> typing.Tuple[int, int]:
		if target[0] == 0:
			return 0, self._window_size
		if target[1] == x.shape[0]:
			return x.shape[0] - self._window_size, x.shape[0]

		return target[0]-self.__padding, target[1]+self.__padding

	def __get_next_target(self, x: np.ndarray, last_target: typing.Union[None, typing.Tuple[int, int]]) -> typing.Tuple[int, int]:
		if last_target is None:
			return 0, self._window_size - self.__padding
		if last_target[-1] + self.__target_size >= x.shape[0]:
			return x.shape[0] - self.__target_size, x.shape[0]
		return last_target[-1], last_target[-1] + self.__target_size

	def __extract_target(self, y_block: np.ndarray, target: typing.Tuple[int, int], source: typing.Tuple[int, int]) -> np.ndarray:
		return y_block[target[0]-source[0]:target[1]-source[0]]

	def __construct_input(self, x: np.ndarray, y: np.ndarray, i: int) -> np.ndarray:
		inputs = np.zeros((1, 2, self._window_size))
		inputs[0, 0] = x
		inputs[0, 1, inputs.shape[-1]-i:] = y[:i]
		return inputs

	def __execute_block(self, x: np.ndarray) -> np.ndarray:
		y = np.zeros(x.shape[0])
		for i in range(y.shape[0]):
			prediction = self._model.predict(self.__construct_input(x, y, i))
			y[i] = prediction.flatten()[0]
		return y

	def _execute(self, X: np.ndarray) -> np.ndarray:

		target: typing.Union[None, typing.Tuple[int, int]] = None
		y = np.zeros(X.shape[0])

		while target is None or target[-1] != X.shape[0]:
			target = self.__get_next_target(X, target)
			source = self.__get_source_block(X, target)
			y_block = self.__execute_block(X[source[0]:source[1]])
			y[target[0]:target[1]] = self.__extract_target(y_block, target, source)

		return y
