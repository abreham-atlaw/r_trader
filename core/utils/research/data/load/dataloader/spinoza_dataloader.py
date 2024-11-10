from abc import ABC, abstractmethod

import torch


class SpinozaDataLoader(ABC):

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def __getitem__(self, idx: int) -> torch.Tensor:
		pass

	@abstractmethod
	def shuffle(self):
		pass
