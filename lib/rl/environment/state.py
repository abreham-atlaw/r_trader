from typing import *
from abc import abstractmethod, ABC

# lib.rl.environment.state.ModelBasedState
class ModelBasedState(ABC):

	@abstractmethod
	def set_depth(self, depth: int):
		pass

	@abstractmethod
	def get_depth(self) -> int:
		pass
