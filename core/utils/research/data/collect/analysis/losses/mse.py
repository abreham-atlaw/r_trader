import numpy as np

from .loss import Loss


class MeanSquaredError(Loss):

	def evaluate(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
		return np.mean((y_hat - y) ** 2)
