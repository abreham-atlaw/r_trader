import typing

import torch

from .data_pool import DataPool


class TensorDataPool(DataPool):

	def shuffle(self):
		pass

	def __init__(self, size: int):
		self.size = size
		self.__is_initialized = False
		self.__pool_idxs = (torch.zeros(self.size, dtype=torch.int32) - 1)	

	def __init_pool(self, X: torch.Tensor, y: torch.Tensor):
		self.__pool_X = torch.zeros((self.size, *X.shape))	
		self.__pool_y = torch.zeros((self.size, *y.shape))	

		self.__pool_idxs = (torch.zeros(self.size, dtype=torch.int32) - 1)	

	def __get_free_slot(self) -> int:
		if not torch.any(self.__pool_idxs == -1):
			self.__pool_idxs[0] = -1

		return list(self.__pool_idxs).index(-1)

	def __get_slot(self, idx: int) -> int:
		return self.__pool_idxs[self.__pool_idxs == idx][0]

	def clean(self):
		self.__pool_idxs = torch.zeros(self.size, dtype=torch.int32) - 1

	def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		if torch.any(self.__pool_idxs == idx):
			slot = self.__get_slot(idx)
			return self.__pool_X[slot], self.__pool_y[slot]

		return None

	def __setitem__(self, key: int, value: typing.Tuple[torch.Tensor, torch.Tensor]):
		if not self.__is_initialized:
			self.__init_pool(value[0], value[1])
		slot = self.__get_free_slot()
		self.__pool_X[slot] = value[0]
		self.__pool_y[slot] = value[1]
		self.__pool_idxs[slot] = key

	def __len__(self):
		return torch.sum(self.__pool_idxs != -1)

	def __contains__(self, idx: int):
		return torch.any(self.__pool_idxs == idx)
