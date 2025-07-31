import typing

import numpy as np

from .splitter import TrainTestSplitter


class SequentialSplitter(TrainTestSplitter):

	def __init__(self, test_size: float = 0.2):
		self.__test_size = test_size

	def split(self, *args: typing.Tuple[np.ndarray, ...]) -> typing.Tuple[typing.Tuple[np.ndarray, ...], typing.Tuple[np.ndarray, ...]]:
		if len(args) < 1:
			raise ValueError("At least one array expected.")

		if False in [len(args[0]) == len(arg) for arg in args]:
			raise ValueError("All inputs should have the same size on axis 0.")

		split_point = int(len(args[0]) * (1-self.__test_size))
		return (
			tuple([arg[:split_point] for arg in args]),
			tuple([arg[split_point:] for arg in args]),
		)
