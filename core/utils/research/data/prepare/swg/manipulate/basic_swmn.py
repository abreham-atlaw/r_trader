import numpy as np

from core.utils.research.data.prepare.swg.manipulate.abstract_swmn import AbstractSampleWeightManipulator


class BasicSampleWeightManipulator(AbstractSampleWeightManipulator):

	def __init__(
			self,
			*args,
			scale: float = 1.0,
			shift: float = 0.0,
			stretch: float = 1.0,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__scale = scale
		self.__shift = shift
		self.__stretch = stretch

	def _manipulate(self, w: np.ndarray) -> np.ndarray:
		return ((w + self.__shift) * self.__scale) ** self.__stretch
