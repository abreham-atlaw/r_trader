from abc import abstractmethod

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator


class BasicSampleWeightGenerator(AbstractSampleWeightGenerator):

	@abstractmethod
	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		pass

	def _generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		return self._generate_weights(X, y)
