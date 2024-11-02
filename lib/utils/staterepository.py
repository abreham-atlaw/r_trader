import json
import typing
from abc import ABC, abstractmethod

import os
import shutil
import pickle

from lib.network.rest_interface import Serializer


class StateRepository(ABC):

	def __init__(self, serializer: Serializer = None):
		self.__serializer = serializer

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

		if self.__serializer is None:
			self.__dump_file(states_map, filepath, False)
			return

		serialized = {
			key: self.__serializer.serialize(state)
			for key, state in states_map.items()
		}

		self.__dump_file(serialized, filepath, True)

	def load(self, filepath: str):

		if self.__serializer is None:
			state_map = self.__load_file(filepath, False)

		else:
			with open(filepath, "r") as f:
				json_ = json.load(f)
			state_map = {
				key: self.__serializer.deserialize(state)
				for key, state in json_.items()
			}

		for key, state in state_map.items():
			self.store(key, state)

	def __len__(self):
		return len(self.get_keys())


class DictStateRepository(StateRepository):

	def __init__(self):
		self.__states = {}

	def store(self, key: str, state):
		self.__states[key] = state

	def retrieve(self, key: str) -> object:
		try:
			return self.__states[key]
		except KeyError:
			raise StateNotFoundException()

	def exists(self, key: str) -> bool:
		return key in self.__states.keys()

	def clear(self):
		self.__states = {}

	def destroy(self):
		del self.__states


class SectionalDictStateRepository(StateRepository):

	def __init__(self, len_: int, depth: int = 1, serializer: Serializer = None):
		super().__init__(serializer)
		self.__len = len_
		self.__depth = depth
		self.__states = {}
		self.__keys = set()

	def __store(self, key: str, state, dict_: dict, depth: int):
		if depth == 0:
			dict_[key] = state
			return
		inner_key = key[:self.__len]
		inner_dict = dict_.get(inner_key)
		if inner_dict is None:
			inner_dict = {}
			dict_[inner_key] = inner_dict

		self.__store(
			key[self.__len:],
			state,
			inner_dict,
			depth-1
		)

	def __retrieve(self, key: str, dict_: dict, depth: int) -> object:
		if depth == 0:
			return dict_[key]
		return self.__retrieve(key[self.__len:], dict_[key[:self.__len]], depth-1)

	def store(self, key: str, state):
		if len(key) <= self.__len * self.__depth:
			raise Exception("Key Too Short. Must be more than %s characters long" % (self.__len * self.__depth, ))
		self.__store(key, state, self.__states, self.__depth)
		self.__keys.add(key)

	def retrieve(self, key: str) -> object:
		try:
			return self.__retrieve(key, self.__states, self.__depth)
		except KeyError:
			raise StateNotFoundException

	def exists(self, key: str) -> bool:
		try:
			self.retrieve(key)
			return True
		except StateNotFoundException:
			return False

	def clear(self):
		self.__states = {}
		self.__keys = set()

	def destroy(self):
		del self.__states
		self.__keys = set()

	def get_keys(self) -> typing.List[str]:
		return list(self.__keys)


class FileSystemStateRepository(StateRepository, ABC):

	def __init__(self, path: str = None, name: str = ".states"):
		if path is None:
			path = os.path.abspath("./")

		self.__container = os.path.join(path, name)
		self.__create_container()

	def __create_container(self):
		if not os.path.isdir(self.__container):
			os.mkdir(self.__container)

	@abstractmethod
	def _store_in(self, state: object, path: str):
		pass

	@abstractmethod
	def _read_from(self, path: str) -> object:
		pass

	def _construct_path(self, key: str) -> str:
		return os.path.join(self.__container, key)

	def store(self, key: str, state):
		state_path = self._construct_path(key)
		if not os.path.isdir(state_path):
			os.mkdir(state_path)
		self._store_in(state, state_path)

	def retrieve(self, key: str) -> object:
		state_path = self._construct_path(key)
		if not os.path.isdir(state_path):
			raise StateNotFoundException()
		return self._read_from(state_path)

	def exists(self, key: str) -> bool:
		return os.path.isdir(self._construct_path(key))

	def clear(self):
		shutil.rmtree(self.__container)
		self.__create_container()

	def destroy(self):
		shutil.rmtree(self.__container)


class PickleStateRepository(FileSystemStateRepository):

	def __init__(self, file_name="content.pkl", **kwargs):
		super().__init__(**kwargs)
		self.__file_name = file_name

	def _construct_file_path(self, path, file_name) -> str:
		return os.path.join(path, file_name)

	def _store_in(self, state: object, path: str):
		with open(self._construct_file_path(path, self.__file_name), "wb") as out_stream:
			pickle.dump(state, out_stream)

	def _read_from(self, path: str) -> object:
		obj = None
		with open(self._construct_file_path(path, self.__file_name), "rb") as in_stream:
			obj = pickle.load(in_stream)

		return obj


class StateNotFoundException(Exception):
	pass
