import typing

import numpy as np

from lib.utils.logger import Logger
from .abstract_swmn import AbstractSampleWeightManipulator


class StandardizeSampleWeightManipulator(AbstractSampleWeightManipulator):

	def __init__(
			self,
			target_std: float,
			target_mean: float,
			current_std: float,
			current_mean: float,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__target_std = target_std
		self.__target_mean = target_mean
		self.__current_std = current_std
		self.__current_mean = current_mean

	def _manipulate(self, w: np.ndarray) -> np.ndarray:
		return ((w - self.__current_mean)*self.__target_std / self.__current_std ) + self.__target_mean
