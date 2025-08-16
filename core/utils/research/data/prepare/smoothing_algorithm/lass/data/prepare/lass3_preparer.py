import numpy as np

from .lass_preparer_2 import LassPreparer2


class Lass3Preparer(LassPreparer2):

	def __init__(
			self,
			*args,
			left_align: bool = False,
			**kwargs
	):
		super().__init__(
			*args,
			**kwargs,
			trim_extra_gran=True
		)
		self.__left_align = left_align

	def __apply_mask(self, x: np.ndarray):
		out = np.repeat(x, x.shape[-1], axis=0)

		encoded_mask = np.arange(
			x.shape[-1]
		).reshape(1, -1) >= (x.shape[-1] - np.tile(np.arange(x.shape[-1]), x.shape[0]).reshape(-1, 1))

		if self.__left_align:
			encoded_mask = np.arange(x.shape[-1]).reshape(1, -1) < (np.tile(np.arange(x.shape[-1]), x.shape[0]).reshape(-1, 1))

		value_mask = np.arange(x.shape[-1]).reshape(1, -1) < np.tile(np.arange(x.shape[-1]), x.shape[0]).reshape(-1, 1)

		out[:, 1][encoded_mask] = out[:, 1][value_mask]
		out[:, 1][~encoded_mask] = 0

		return out

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		x = self._stack_noisy_and_smoothed(sequences)
		x = self.__apply_mask(x)
		return x

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		data = self._stack_noisy_and_smoothed(sequences)
		smoothed = data[:, 1, :]
		y = smoothed.reshape(-1, 1)
		return y
