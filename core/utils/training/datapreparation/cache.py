from typing import *

import gc


class Cache:

	def __init__(self, size: int):
		self.__size = size
		self.__order = []
		self.__cache = {}

	def store(self, key, value):
		if len(self.__order) == self.__size:
			del self.__cache[self.__order.pop(0)]
			gc.collect()
		self.__order.append(key)
		self.__cache[key] = value

	def retrieve(self, key) -> Optional[object]:
		return self.__cache.get(key)
