import typing
from abc import ABC, abstractmethod

import torch


class DataPool(ABC):

	@abstractmethod
	def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		pass

	@abstractmethod
	def __setitem__(self, key: int, value: typing.Tuple[torch.Tensor, torch.Tensor]):
		pass

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def __contains__(self, idx: int) -> bool:
		pass

	@abstractmethod
	def clean(self):
		pass

	@abstractmethod
	def shuffle(self):
		pass
