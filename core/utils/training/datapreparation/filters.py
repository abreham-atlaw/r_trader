from typing import *
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras.layers import Layer

from lib.dnn.layers import Norm, MovingAverage, KalmanFilter, WeightedMovingAverage, OverlayIndicator


class Filter(ABC):

	def calc_input_shape(self, output_shape: Tuple) -> Tuple:
		return output_shape

	@abstractmethod
	def filter(self, inputs: np.ndarray) -> np.ndarray:
		pass


class LayerFilter(Filter, ABC):

	def __init__(self, layer: Layer):
		self._layer = layer

	def filter(self, inputs: np.ndarray) -> np.ndarray:
		return self._layer(inputs).numpy()


class OverlayFilter(LayerFilter):

	def __init__(self, layer: OverlayIndicator):
		super().__init__(layer)

	def calc_input_shape(self, output_shape: Tuple) -> Tuple:
		return output_shape[0], output_shape[1] + self._layer.get_window_size() - 1


class NormFilter(LayerFilter):

	def __init__(self):
		super().__init__(layer=Norm())


class MovingAverageFilter(OverlayFilter):

	def __init__(self, window_size):
		super().__init__(MovingAverage(window_size=window_size))


class KalmanFilterFilter(LayerFilter):

	def __init__(self, alpha: float, beta: float):
		super().__init__(KalmanFilter(alpha, beta))


class WeightedMovingAverageFilter(OverlayFilter):

	def __init__(self, window_size: int):
		super().__init__(WeightedMovingAverage(window_size=window_size))
