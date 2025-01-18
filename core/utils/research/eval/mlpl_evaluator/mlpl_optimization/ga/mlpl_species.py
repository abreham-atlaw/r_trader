import random
import typing

from dataclasses import dataclass
from typing import List

from core.utils.research.model.model.savable import SpinozaModule
from lib.ga import Species, GAUtils
from lib.ga.choice_utils import ChoiceUtils


@dataclass
class MLPLSpecies(Species):

	models: typing.List[str]

	def mutate(self, *args, **kwargs):
		self.models.pop(random.randint(0, len(self.models) - 1))

	def reproduce(self, spouse: 'MLPLSpecies', preferred_offsprings: int) -> List['MLPLSpecies']:
		return [
			MLPLSpecies(
				models=list(set(ChoiceUtils.list_select(
					self.models,
					spouse.models,
					discrete=True,
					size_noise=0.5
				)))
			)
			for _ in range(preferred_offsprings)
		]
