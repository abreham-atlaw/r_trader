import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from core.utils.research.data.prepare.time_series_data_preparer import TimeSeriesDataPreparer
from lib.utils.logger import Logger


class LassPreparer(TimeSeriesDataPreparer):

	def __init__(
			self,
			sa: SmoothingAlgorithm,
			shift: int,
			block_size: int,
			*args,
			**kwargs
	):
		self._shift = shift
		self._sa = sa
		super().__init__(
			*args,
			block_size=self._get_input_block_size(block_size),
			**kwargs
		)
		Logger.info(f"Using Smoothing Algorithm: {self._sa}")

	def _get_input_block_size(self, block_size: int) -> int:
		return block_size + self._shift

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		return sequences[:, :-self._shift]

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		smoothed_sequence = self._sa.apply_on_batch(sequences)
		return smoothed_sequence[:, -1:]
