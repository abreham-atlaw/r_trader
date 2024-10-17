from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):

	@abstractmethod
	def evaluate(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
		pass
