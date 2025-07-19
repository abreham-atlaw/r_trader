import typing
from abc import ABC, abstractmethod

import numpy as np


class TrainTestSplitter(ABC):

	@abstractmethod
	def split(self, *args: typing.Tuple[np.ndarray, ...]) -> typing.Tuple[typing.Tuple[np.ndarray, ...], typing.Tuple[np.ndarray, ...]]:
		pass

	def __str__(self):
		return f"{self.__class__.__name__}"
