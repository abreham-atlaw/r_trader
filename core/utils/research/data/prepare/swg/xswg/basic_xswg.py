from abc import abstractmethod, ABC

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator


class BasicXSampleWeightGenerator(AbstractSampleWeightGenerator, ABC):

	@abstractmethod
	def _generate(self, X: np.ndarray) -> np.ndarray:
		pass
