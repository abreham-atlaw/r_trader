import typing

import numpy as np

from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from .msmm import MarketStateMemoryMatcher


class BoundMarketStateMemoryMatcher(MarketStateMemoryMatcher):

	def __init__(self, *args, bounds: typing.Iterable, **kwargs):
		super().__init__(*args, **kwargs)
		if not isinstance(bounds, np.ndarray):
			bounds = np.array(bounds)
		self.__bounds = bounds

	@staticmethod
	def __get_percentages(sequence: np.ndarray) -> np.ndarray:
		return sequence[1:] / sequence[:-1]

	def __get_bounds(self, percentages: np.ndarray) -> np.ndarray:
		return np.array([
			DataPrepUtils.find_bound_index(self.__bounds, p)
			for p in percentages
		])

	def _compare_instrument_state(self, cue: np.ndarray, memory: np.ndarray) -> bool:
		cue_percentages, memory_percentages = self.__get_percentages(cue), self.__get_percentages(memory)
		cue_bounds, memory_bounds = self.__get_bounds(cue_percentages), self.__get_bounds(memory_percentages)
		return np.all(cue_bounds == memory_bounds)
