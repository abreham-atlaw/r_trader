import numpy as np

from core.utils.research.data.prepare.swg.manipulate import AbstractSampleWeightManipulator


class ReciprocateSampleWeightManipulator(AbstractSampleWeightManipulator):

	def __init__(
			self,
			*args,
			epsilon: float = 1e-9,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__epsilon = epsilon

	def _manipulate(self, w: np.ndarray) -> np.ndarray:
		return 1/(w + self.__epsilon)

