from typing import *
from abc import ABC, abstractmethod

import hashlib
import gc


class Cache(ABC):

	def _prepare_key(self, key: str) -> Hashable:
		return hashlib.md5(bytes(key, encoding="utf-8")).hexdigest()

	def store(self, key: str, value: float):
		self._store(self._prepare_key(key), value)

	@abstractmethod
	def _store(self, key: Hashable, value: float):
		pass

	def retrieve(self, key: str) -> Optional[float]:
		return self._retrieve(self._prepare_key(key))

	@abstractmethod
	def _retrieve(self, key: Hashable):
		pass

	@abstractmethod
	def clear(self):
		pass



class HashMapCache(Cache):

	def __init__(self):
		self.__cache = {}

	def _store(self, key: Hashable, value: float):
		self.__cache[key] = value

	def _retrieve(self, key: Hashable) -> Optional[float]:
		return self.__cache.get(key)

	def clear(self):
		self.__cache.clear()
		gc.collect()



class DataRepository(ABC, Sized):

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

	@abstractmethod
	def reset(self):
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

	def reset(self):
		self.__request_queue = []
		self.__response_map = {}

	def __len__(self):
		return len(self.__request_queue)
