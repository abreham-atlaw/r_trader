import json
import typing
from abc import ABC, abstractmethod

import os
import shutil
import pickle

from lib.network.rest_interface import Serializer


class StateRepository(ABC):

	def __init__(self, serializer: Serializer = None):
		self._serializer = serializer

	@abstractmethod
	def store(self, key: str, state):
		pass

	@abstractmethod
	def retrieve(self, key: str) -> object:
		pass

	@abstractmethod
	def exists(self, key: str) -> bool:
		pass

	@abstractmethod
	def clear(self):
		pass

	@abstractmethod
	def destroy(self):
		pass

	@abstractmethod
	def get_keys(self) -> typing.List[str]:
		pass

	def remove(self, key: str):
		self.store(key, None)

	def __get_all(self):
		return {
			key: self.retrieve(key)
			for key in self.get_keys()
		}

	@staticmethod
	def __dump_file(content, filepath: str, use_json: bool):
		if use_json:
			with open(filepath, "w") as f:
				json.dump(content, f)
			return

		with open(filepath, "wb") as f:
			pickle.dump(content, f)

	@staticmethod
	def __load_file(filepath: str, use_json: bool):
		if use_json:
			with open(filepath, "r") as f:
				return json.load(f)

		with open(filepath, "rb") as f:
			return pickle.load(f)

	def dump(self, filepath: str, keys: typing.List[str] = None):

		if keys is None:
			states_map = self.__get_all()
		else:
			states_map = {
				key: self.retrieve(key)
				for key in keys
			}

		if self._serializer is None:
			self.__dump_file(states_map, filepath, False)
			return

		serialized = {
			key: self._serializer.serialize(state)
			for key, state in states_map.items()
		}

		self.__dump_file(serialized, filepath, True)

	def load(self, filepath: str):

		if self._serializer is None:
			state_map = self.__load_file(filepath, False)

		else:
			with open(filepath, "r") as f:
				json_ = json.load(f)
			state_map = {
				key: self._serializer.deserialize(state)
				for key, state in json_.items()
			}

		for key, state in state_map.items():
			self.store(key, state)

	def __len__(self):
		return len(self.get_keys())


class StateNotFoundException(Exception):
	pass
