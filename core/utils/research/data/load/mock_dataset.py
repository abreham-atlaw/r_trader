import typing

import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):

	def __init__(self, size: int, shapes: typing.List[typing.Tuple[int, ...]], dtype=torch.float32):
		super().__init__()
		self.__size = size
		self.__shape = shapes
		self.__dtype = dtype

	def __len__(self):
		return self.__size

	def __getitem__(self, item):
		return [
			torch.rand(shape, dtype=self.__dtype)
			for shape in self.__shape
		]

	def shuffle(self):
		pass
