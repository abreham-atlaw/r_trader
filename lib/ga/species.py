from typing import *
from abc import ABC, abstractmethod

import random


class Species(ABC):

	@abstractmethod
	def mutate(self, *args, **kwargs):
		pass

	def reproduce(self, spouse: 'Species', preferred_offsprings: int) -> List['Species']:
		pass

	def choose_spouse(self, options: List['Species']) -> 'Species':
		return max(options, key=self._evaluate_spouse_compatibility)

	def _evaluate_spouse_compatibility(self, other: 'Species') -> float:
		return random.random()


class ClassDictSpecies(Species, ABC):

	def __init__(self):
		self.__gene_classes = [ClassDictSpecies] + self._get_gene_classes()

	@abstractmethod
	def _get_gene_classes(self):
		pass

	def _get_excluded_values(self) -> List[str]:
		return [
			key
			for key in list(self.__dict__)
			if key.startswith("_")
		]

	@staticmethod
	def __get_random_neighbor(x: int) -> int:
		return round((random.random() + 0.5)*x)

	@staticmethod
	def __get_random_mean(x0, x1, expand_bounds: bool = True, non_negative=True) -> int:
		if expand_bounds:
			x0, x1 = (5 * x0 - x1) / 4, (5 * x1 - x0) / 4
		if non_negative:
			x0, x1 = max(x0, 0), max(x1, 0)
		w = random.random()
		return round((x0 * w) + ((1 - w) * x1))

	def __mutate_gene(self, gene: Any) -> Any:

		if isinstance(gene, List):
			if len(gene) == 0:
				return gene  # TODO
			for _ in range(random.randint(0, len(gene))):
				index = random.randint(0, len(gene) - 1)
				gene[index] = self.__mutate_gene(gene[index])
			return gene

		if isinstance(gene, int):
			return self.__get_random_neighbor(gene)

		if isinstance(gene, bool):
			return random.choice([True, False])

		if isinstance(gene, tuple(self.__gene_classes)):

			for _ in range(random.randint(1, len(gene.__dict__))):
				key = random.choice([key for key in list(gene.__dict__.keys()) if key not in self._get_excluded_values()])
				gene.__dict__[key] = self.__mutate_gene(gene.__dict__[key])
			return gene

		return gene  # TODO: CREATE A LIST OF EQUIVALENT GENES

	def __select_gene(self, self_value, spouse_value):

		if isinstance(self_value, List):
			swap_size = min(len(self_value), len(spouse_value))
			new_value = [self.__select_gene(self_value[i], spouse_value[i]) for i in range(swap_size)]
			length = self.__get_random_mean(len(self_value), len(spouse_value))
			if length > len(new_value) and max(len(self_value), len(spouse_value)) != 0:
				larger_genes = self_value
				if len(spouse_value) > len(self_value):
					larger_genes = spouse_value
				if len(larger_genes) < length:
					larger_genes.extend([
						self.__select_gene(
							random.choice(self_value),
							random.choice(spouse_value)
						)
						if swap_size != 0
						else random.choice(larger_genes)
						for i in range(length - len(larger_genes))
					])
				new_value.extend(larger_genes[len(new_value): length])
			return new_value

		if isinstance(self_value, int) and not isinstance(self_value, bool):
			return ClassDictSpecies.__get_random_mean(self_value, spouse_value)

		if isinstance(self_value, tuple(self.__gene_classes)):
			return self_value.__class__(**{
				key: self.__select_gene(self_value.__dict__[key], spouse_value.__dict__[key])
				for key in self_value.__dict__.keys()
				if key not in self._get_excluded_values()
			})

		return random.choice((self_value, spouse_value))

	def _generate_offspring(self, spouse) -> Species:
		return self.__select_gene(self, spouse)

	def reproduce(self, spouse: 'Species', preferred_offsprings: int) -> List['Species']:
		return [
			self._generate_offspring(spouse)
			for _ in range(preferred_offsprings)
		]

	def mutate(self, *args, **kwargs):
		return self.__mutate_gene(self)
