import os
import typing
from abc import ABC, abstractmethod

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from lib.utils.logger import Logger


class AbstractSampleWeightMerger(AbstractSampleWeightGenerator):

	@abstractmethod
	def _merge(self, weights: typing.List[np.ndarray]) -> np.ndarray:
		pass

	def _generate(self, *args, ) -> np.ndarray:
		return self._merge(list(args))
