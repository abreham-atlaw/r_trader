from typing import Iterable

import numpy as np


class Cache:

	def __init__(self):
		self.__store = {}

	def __hash(self, value) -> int:
		return hash(value)

	def store(self, key, value):
		self.__store[self.__hash(key)] = value

	def retrieve(self, key):
		return self.__store.get(self.__hash(key))

	def cached_or_execute(self, key, func):
		value = self.retrieve(key)
		if value is None:
			value = func()
			self.store(key, value)
		return value

	def remove(self, key):
		self.__store.pop(self.__hash(key), None)