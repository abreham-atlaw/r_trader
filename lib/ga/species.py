from typing import *
from abc import ABC, abstractmethod

import random


class Species(ABC):

	@abstractmethod
	def mutate(self, *args, **kwargs):
		pass

	@abstractmethod
	def reproduce(self, spouse: 'Species', preferred_offsprings: int) -> List['Species']:
		pass

	def choose_spouse(self, options: List['Species']) -> 'Species':
		return max(options, key=self._evaluate_spouse_compatibility)

	def _evaluate_spouse_compatibility(self, other: 'Species') -> float:
		return random.random()
