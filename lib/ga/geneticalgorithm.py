from typing import *
from abc import ABC, abstractmethod

import random
import math

from .species import Species
from .callbacks import Callback


class GeneticAlgorithm(ABC):

	def __init__(self, generation_growth_factor=1, mutation_rate=0.3):
		self.__generation_growth_factor = generation_growth_factor
		self.__mutation_rate = mutation_rate

	@abstractmethod
	def _generate_initial_generation(self) -> List[Species]:
		pass

	@abstractmethod
	def _evaluate_species(self, species: Species) -> float:
		pass

	def _get_preferred_offsprings(self, population_size: int) -> int:
		return math.ceil((self.__generation_growth_factor * 10))

	def _choose_primary_spouse(self, population: List[Species]) -> Species:
		return random.choice(population)

	def _filter_generation(self, population: List[Species], target_size: int) -> List[Species]:
		return sorted(population, key=lambda species: self._evaluate_species(species), reverse=True)[:target_size]

	def _mutate_population(self, population: List[Species]):
		for species in population:
			if random.random() < self.__mutation_rate:
				species.mutate()

	def _match_spouses(self, population: List[Species]) -> List[Tuple[Species, Species]]:

		if len(population) == 2:
			return [(population[0], population[1])]

		if len(population) == 1:
			return [(population[0], population[0])]

		wife = self._choose_primary_spouse(population)
		husband = wife.choose_spouse([species for species in population if species != wife])
		return [(wife, husband)] + self._match_spouses([
															species
															for species in population
															if species != wife and species != husband
														])

	def _generate_generation(self, parent_generation: List[Species]) -> List[Species]:
		parents = self._match_spouses(parent_generation)
		new_generation = []
		for mother, father in parents:
			new_generation.extend(mother.reproduce(father, self._get_preferred_offsprings(len(parent_generation)) ))

		return new_generation

	def _render(self, population):

		for species in population:
			print(species)

		print("\n"*2, "-"*100)

	@staticmethod
	def __perform_callback(callbacks: List[Callback], population: List[Species], start):
		for callback in callbacks:
			fn = callback.on_epoch_end
			if start:
				fn = callback.on_epoch_start
			fn(population)

	def __perform_epochs(self, parent_generation: List[Species], epochs: int, callbacks: List[Callback]) -> List[Species]:
		print("Epochs Left: %s" % (epochs,))

		self.__perform_callback(callbacks, parent_generation, True)

		self._mutate_population(parent_generation)
		new_generation = self._generate_generation(parent_generation)
		new_generation_size = int(len(parent_generation) * self.__generation_growth_factor)
		new_generation = self._filter_generation(new_generation, new_generation_size)
		self._render(new_generation)

		self.__perform_callback(callbacks, new_generation, False)

		if epochs == 1:
			return new_generation
		return self.__perform_epochs(new_generation, epochs - 1)

	def start(self, epochs: int, callbacks: Optional[List[Callback]]=None) -> List[Tuple[Species, float]]:

		if callbacks is None:
			callbacks = []

		initial_generation = self._generate_initial_generation()
		final_generation = self.__perform_epochs(initial_generation, epochs, callbacks)

		return [(species, self._evaluate_species(species)) for species in final_generation]
