from typing import *
from abc import ABC, abstractmethod

import random
import math

from lib.utils.logger import Logger
from .species import Species
from .callbacks import Callback


class GeneticAlgorithm(ABC):

	def __init__(self, generation_growth_factor=1, mutation_rate=0.3, preferred_offsprings=None):
		self.__generation_growth_factor = generation_growth_factor
		self.__mutation_rate = mutation_rate
		self.__preferred_offsprings = preferred_offsprings
		self.__loaded_initial_generation = None

	def set_initial_generation(self, generation: List[Species]):
		self.__loaded_initial_generation = generation

	@abstractmethod
	def _generate_initial_generation(self) -> List[Species]:
		pass

	@abstractmethod
	def _evaluate_species(self, species: Species) -> float:
		pass

	def _get_preferred_offsprings(self, population_size: int) -> int:
		if self.__preferred_offsprings is not None:
			return self.__preferred_offsprings
		return math.ceil((self.__generation_growth_factor * 5))

	def _choose_primary_spouse(self, population: List[Species]) -> Species:
		return random.choice(population)

	def _filter_generation(self, population: List[Species], target_size: int) -> List[Species]:
		Logger.info(f"Filtering Population Size: {len(population)} => {target_size}")
		return sorted(population, key=lambda species: self._evaluate_species(species), reverse=True)[:target_size]

	def _mutate_population(self, population: List[Species]):
		for species in population:
			if random.random() < self.__mutation_rate:
				species.mutate()

	def _match_spouses(self, population: List[Species]) -> List[Tuple[Species, Species]]:

		if len(population) == 0:
			return []

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
			new_generation.extend(mother.reproduce(father, self._get_preferred_offsprings(len(parent_generation))))

		return new_generation

	def _render(self, population):

		population_values = [self._evaluate_species(species) for species in population]

		log = f"\n\nPopulation Average Value: {sum(population_values)/len(population)}\n\n"

		for species, value in zip(population, population_values):
			log += str(species) + f"\nValue: {value}"

		log += "\n"*2 + "-"*100

		Logger.info(log)

	def __get_initial_generation(self) -> List[Species]:
		if self.__loaded_initial_generation is None:
			return self._generate_initial_generation()
		return self.__loaded_initial_generation

	@staticmethod
	def __perform_callback(callbacks: List[Callback], population: List[Species], start):
		for callback in callbacks:
			fn = callback.on_epoch_end
			if start:
				fn = callback.on_epoch_start
			fn(population)

	def _perform_epoch(self, parent_generation: List[Species], callbacks: List[Callback]) -> List[Species]:

		self.__perform_callback(callbacks, parent_generation, True)

		self._mutate_population(parent_generation)
		new_generation = parent_generation + self._generate_generation(parent_generation)
		new_generation_size = int(len(parent_generation) * self.__generation_growth_factor)
		new_generation = self._filter_generation(new_generation, new_generation_size)
		self._render(new_generation)

		self.__perform_callback(callbacks, new_generation, False)

		return new_generation

	def start(self, epochs: int, callbacks: Optional[List[Callback]] = None) -> List[Species]:

		if callbacks is None:
			callbacks = []

		generation = self.__get_initial_generation()

		for epoch in range(epochs):
			Logger.info(f"Epoch: {epoch+1}/{epochs}\t\tPopulation Size: {len(generation)}")
			generation = self._perform_epoch(generation, callbacks)
		Logger.info("Done")
		return generation
