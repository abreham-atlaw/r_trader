import typing

import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):

	def __init__(self, X_block_size: int, y_block_size: int, size: int):
		self.__X_block_size = X_block_size
		self.__y_block_size = y_block_size
		self.__size = size

	def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		return tuple([torch.randn(size) for size in [self.__X_block_size, self.__y_block_size]])

	def __len__(self):
		return self.__size

	def shuffle(self):
		pass
