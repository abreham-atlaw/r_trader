import typing

import numpy as np

from .abstract_swm import AbstractSampleWeightMerger


class MeanSampleWeightMerger(AbstractSampleWeightMerger):

	def _merge(self, weights: typing.List[np.ndarray]) -> np.ndarray:
		return np.mean(weights, axis=0)
