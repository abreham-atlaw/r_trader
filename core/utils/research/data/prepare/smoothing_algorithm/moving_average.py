import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm.smoothing_algorithm import SmoothingAlgorithm
from lib.utils.math import moving_average


class MovingAverage(SmoothingAlgorithm):

	def __init__(self, window_size: int):
		self.__window_size = window_size

	def apply(self, x: np.ndarray) -> np.ndarray:
		return moving_average(x, self.__window_size)
