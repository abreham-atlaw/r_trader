import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from core.utils.research.data.prepare.time_series_data_preparer import TimeSeriesDataPreparer


class LassPreparer(TimeSeriesDataPreparer):

	def __init__(
			self,
			sa: SmoothingAlgorithm,
			shift: int,
			block_size: int,
			*args,
			**kwargs
	):
		super().__init__(
			*args,
			block_size=block_size+shift,
			**kwargs
		)
		self.__shift = shift
		self.__sa = sa

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		return sequences[:, :-self.__shift]

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		smoothed_sequence = self.__sa.apply_on_batch(sequences)
		return smoothed_sequence[:, -1:]
