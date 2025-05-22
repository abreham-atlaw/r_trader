from abc import ABC, abstractmethod

import numpy as np

import os

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from lib.utils.logger import Logger


class AbstractSampleWeightManipulator(AbstractSampleWeightGenerator):

	@abstractmethod
	def _manipulate(self, w: np.ndarray) -> np.ndarray:
		pass

	def _generate(self, w: np.ndarray) -> np.ndarray:
		return self._manipulate(w)
