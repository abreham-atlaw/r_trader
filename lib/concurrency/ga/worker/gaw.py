from typing import *
from abc import abstractmethod, ABC

import time
import os
from requests.exceptions import HTTPError

from lib.utils.logger import Logger
from lib.ga import Species, GeneticAlgorithm
from lib.network.rest_interface import NetworkApiClient, Serializer
from .requests import EvaluateRequest, ResponseRequest, EvaluateResponse


class GAWorker(GeneticAlgorithm, ABC):

	def __init__(self, server_url: str, sleep_time=5, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__network_client = NetworkApiClient(os.path.join(server_url, "worker"))
		self.__serializer = self._init_serializer()
		self.__sleep_time = sleep_time

	@abstractmethod
	def _init_serializer(self) -> Serializer:
		pass

	def _generate_initial_generation(self) -> List[Species]:
		pass

	def __get_request(self) -> Tuple[str, Species]:
		while True:
			try:
				response: Optional[EvaluateResponse] = self.__network_client.execute(EvaluateRequest())
				return response.key, self.__serializer.deserialize(response.species)
			except HTTPError:
				pass
			time.sleep(self.__sleep_time)

	def __process_request(self):
		key, species = self.__get_request()
		evaluation = self._evaluate_species(species)
		self.__network_client.execute(ResponseRequest(key, evaluation))

	def start(self, *args, **kwargs):
		while True:
			try:
				self.__process_request()
			except Exception as ex:
				print(f"Failed to Process Request: {ex}")

