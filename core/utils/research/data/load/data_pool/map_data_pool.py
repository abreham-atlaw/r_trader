import typing
from collections import OrderedDict

import torch

from .data_pool import DataPool


class MapDataPool(DataPool):

	def __init__(self, size: int):
		self.__size = size
		self.__pool = OrderedDict()

	def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		return self.__pool.get(idx)

	def __setitem__(self, key: int, value: typing.Tuple[torch.Tensor, torch.Tensor]):
		self.__pool[key] = value
		if len(self.__pool) > self.__size:
			self.__pool.popitem(last=False)

	def __len__(self) -> int:
		return len(self.__pool)

	def __contains__(self, idx: int) -> bool:
		return idx in self.__pool

	def clean(self):
		self.__pool = OrderedDict()

	def shuffle(self):
		pass
