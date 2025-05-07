import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm.smoothing_algorithm import SmoothingAlgorithm
from lib.utils.math import kalman_filter


class KalmanFilter(SmoothingAlgorithm):

	def __init__(self, alpha: float, beta: float):
		self.__alpha, self.__beta = alpha, beta

	def apply(self, x: np.ndarray) -> np.ndarray:
		return kalman_filter(x, self.__alpha, self.__beta)

	def __str__(self):
		return f"KalmanFilter({self.__alpha}, {self.__beta})"
