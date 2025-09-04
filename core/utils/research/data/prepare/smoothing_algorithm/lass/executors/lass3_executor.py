import typing

import numpy as np

from core.utils.research.model.model.savable import SpinozaModule
from .lass_executor import LassExecutor


class Lass3Executor(LassExecutor):

	def __init__(
			self,
			*args,
			padding: int = 0,
			left_align: bool = False,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._padding = padding
		self.__target_size = None
		self._left_align = left_align

	def set_model(self, model: SpinozaModule):
		super().set_model(model)
		self.__target_size = self._window_size - 2 * self._padding

	def _get_source_block(self, x: np.ndarray, target: typing.Tuple[int, int]) -> typing.Tuple[int, int]:
		if target[0] == 0:
			return 0, self._window_size
		if target[1] + self._padding >= x.shape[0]:
			return x.shape[0] - self._window_size, x.shape[0]

		return target[0] - self._padding, target[1] + self._padding

	def _get_next_target(self, x: np.ndarray, last_target: typing.Union[None, typing.Tuple[int, int]]) -> typing.Tuple[int, int]:
		if last_target is None:
			return 0, self._window_size - self._padding
		if last_target[-1] + self.__target_size >= x.shape[0]:
			return x.shape[0] - self.__target_size, x.shape[0]
		return last_target[-1], last_target[-1] + self.__target_size

	def __extract_target(self, y_block: np.ndarray, target: typing.Tuple[int, int], source: typing.Tuple[int, int]) -> np.ndarray:
		return y_block[target[0]-source[0]:target[1]-source[0]]

	def __construct_input(self, x: np.ndarray, y: np.ndarray, i: int) -> np.ndarray:
		inputs = np.zeros((1, 2, self._window_size))
		inputs[0, 0] = x
		if not self._left_align:
			inputs[0, 1, inputs.shape[-1]-i:] = y[:i]
		else:
			inputs[0, 1, :i] = y[:i]
		return inputs

	def _process_prediction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		return y.flatten()[0]

	def __execute_block(self, x: np.ndarray, y: np.ndarray, start: int) -> np.ndarray:
		y = y.copy()
		for i in range(start, y.shape[0]):
			inputs = self.__construct_input(x, y, i)
			prediction = self._model.predict(inputs)
			y[i] = self._process_prediction(inputs, prediction)
		return y

	def _init_y(self, X: np.ndarray) -> np.ndarray:
		return np.zeros(X.shape[0])

	def _execute(self, X: np.ndarray) -> np.ndarray:

		target: typing.Union[None, typing.Tuple[int, int]] = None
		y = self._init_y(X)

		while target is None or target[-1] != X.shape[0]:
			target = self._get_next_target(X, target)
			source = self._get_source_block(X, target)
			y_block = self.__execute_block(
				X[source[0]:source[1]],
				y[source[0]:source[1]],
				target[0]-source[0]
			)
			y[target[0]:target[1]] = self.__extract_target(y_block, target, source)

		return y
