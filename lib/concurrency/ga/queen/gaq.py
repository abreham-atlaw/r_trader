from typing import *
from abc import ABC, abstractmethod

import time
import hashlib
import os
from requests.exceptions import HTTPError

from lib.ga import GeneticAlgorithm, Species
from lib.network.rest_interface import Serializer, NetworkApiClient
from .requests import EvaluateRequest, GetResult


class GAQueen(GeneticAlgorithm, ABC):

	def __init__(self, server_address, *args, sleep_time=5,  **kwargs):
		super().__init__(*args, **kwargs)
		self.__network_client = NetworkApiClient(os.path.join(server_address, "queen"))
		self.__species_serializer = self._init_serializer()
		self.__sleep_time = sleep_time

	@abstractmethod
	def _init_serializer(self) -> Serializer:
		pass

	def __collect_result(self, key: str) -> Optional[float]:
		try:
			return self.__network_client.execute(GetResult(key))
		except HTTPError:
			return None

	def __collect_results(self, keys: List[str]) -> List[float]:
		values = [None for _ in keys]
		while None in values:

			for i, key in enumerate(keys):
				values[i] = self.__collect_result(key)
			time.sleep(self.__sleep_time)

		return values

	def __create_requests(self, keys: List[str], population: List[Species]):
		for key, species in zip(keys, population):
			self.__network_client.execute(
				EvaluateRequest(
					key,
					self.__species_serializer.serialize(species)
				)
			)

	def _generate_key(self, species: Species, index: int) -> str:
		return hashlib.md5(
			bytes(
				f"{index}-{self.__species_serializer.serialize_json(species)}",
				encoding="utf-8"
			)
		).hexdigest()

	def _filter_generation(self, population: List[Species], target_size: int) -> List[Species]:
		keys = [self._generate_key(species, i) for i, species in enumerate(population)]
		self.__create_requests(keys, population)
		values = self.__collect_results(keys)
		return sorted(population, key=lambda species: values[population.index(species)], reverse=True)[:target_size]

	def _evaluate_species(self, species: Species) -> float:
		pass
