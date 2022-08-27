from typing import *
from abc import abstractmethod, ABC


class ModelBasedState(ABC):

	@abstractmethod
	def set_depth(self, depth: int):
		pass

	@abstractmethod
	def get_depth(self) -> int:
		pass
