from typing import *
from abc import ABC, abstractmethod

import numpy as np

import time
import hashlib
import os
from datetime import datetime
from requests.exceptions import HTTPError

from lib.utils.logger import Logger
from lib.ga import GeneticAlgorithm, Species
from lib.network.rest_interface import Serializer, NetworkApiClient
from lib.utils.tuplemap import TupleMap
from .requests import EvaluateRequest, GetResult, ResetRequest


class GAQueen(GeneticAlgorithm, ABC):

	def __init__(self, server_address, *args, sleep_time=5, timeout=12*60*60, default_value=None,  **kwargs):
		super().__init__(*args, **kwargs)
		self.__network_client = NetworkApiClient(os.path.join(server_address, "queen"))
		self.__species_serializer = self._init_serializer()
		self.__sleep_time = sleep_time
		self.__timeout = timeout
		self.__default_value = None
		self.__values_map = TupleMap()

	@abstractmethod
	def _init_serializer(self) -> Serializer:
		pass

	def __get_default_value(self) -> float:
		if self.__default_value is None:
			return -1*np.random.random()
		return self.__default_value

	def __collect_result(self, key: str) -> Optional[float]:
		try:
			return self.__network_client.execute(GetResult(key))
		except HTTPError:
			return None

	def __collect_results(self, keys: List[str]) -> List[float]:
		values = [None for _ in keys]
		start_datetime = datetime.now()
		while None in values:

			for i, key in enumerate(keys):
				if values[i] is None:
					values[i] = self.__collect_result(key)

			if (datetime.now() - start_datetime).total_seconds() >= self.__timeout:
				Logger.info(f"Timeout. Complete: {len([value for value in values if value is not None])}/{len(values)}. Filling values.")
				break

			Logger.info(f"Time left: {self.__timeout - (datetime.now() - start_datetime).total_seconds(): .2f}s", end="\r")
			time.sleep(self.__sleep_time)

		for i in range(len(values)):
			if values[i] is None or np.isnan(values[i]):
				values[i] = self.__get_default_value()

		return values

	def __create_requests(self, keys: List[str], population: List[Species]):
		for key, species in zip(keys, population):
			self.__network_client.execute(
				EvaluateRequest(
					key,
					self.__species_serializer.serialize(species)
				)
			)

	def __remote_evaluate(self, population: List[Species]) -> List[float]:
		keys = [self._generate_key(species, i) for i, species in enumerate(population)]
		self.__create_requests(keys, population)
		return self.__collect_results(keys)

	def __store_values(self, population, values):
		self.__values_map.clear()
		for instance, value in zip(population, values):
			self.__values_map.store(instance, value)

	def _generate_key(self, species: Species, index: int) -> str:
		return hashlib.md5(
			bytes(
				f"{self.__species_serializer.serialize_json(species)}",
				encoding="utf-8"
			)
		).hexdigest()

	def _filter_generation(self, population: List[Species], target_size: int) -> List[Species]:
		self.__store_values(population, self.__remote_evaluate(population))
		return super()._filter_generation(population, target_size)

	def _perform_epoch(self, *args, **kwargs) -> List[Species]:
		self.__network_client.execute(
			ResetRequest()
		)
		return super()._perform_epoch(*args, **kwargs)

	def _evaluate_species(self, species: Species) -> float:
		value = self.__values_map.retrieve(species)
		if value is None:
			raise ValueError("Value not evaluated.")
		return value
