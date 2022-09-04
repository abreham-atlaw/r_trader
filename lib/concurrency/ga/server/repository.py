from typing import *
from abc import ABC, abstractmethod


class DataRepository:

	@abstractmethod
	def add_to_request_queue(self, key: str, request: object):
		pass

	@abstractmethod
	def get_request(self) -> Tuple[str, object]:
		pass

	@abstractmethod
	def set_response(self, key: str, value: float):
		pass

	@abstractmethod
	def get_response(self, key: str) -> float:
		pass


class PlainDataRepository(DataRepository):

	def __init__(self):
		self.__request_queue: List[Tuple[str, object]] = []
		self.__response_map: Dict[str, float] = {}

	def add_to_request_queue(self, key: str, request: object):
		self.__request_queue.append((key, request))

	def get_request(self) -> Optional[Tuple[str, object]]:
		if len(self.__request_queue) == 0:
			return None
		return self.__request_queue.pop(0)

	def set_response(self, key: str, value: float):
		self.__response_map[key] = value

	def get_response(self, key: str) -> Optional[float]:
		return self.__response_map.get(key)
