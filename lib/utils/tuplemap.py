from typing import *


class TupleMap:

	def __init__(self):
		self.__values = []

	def store(self, key: Any, value: Any):
		self.remove(key)
		self.__values.append((key, value))

	def retrieve(self, key: Any) -> Optional[Any]:
		for stored_key, stored_value in self.__values:
			if stored_key == key:
				return stored_value
		return None

	def remove(self, key):
		for i, stored_key in enumerate(self.__values):
			if stored_key == key:
				self.__values = self.__values[:i] + self.__values[i+1:]
				break

	def clear(self):
		self.__values = []
