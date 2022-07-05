from abc import ABC, abstractmethod

import os
import shutil
import pickle


class StateRepository(ABC):

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

	def __init__(self, len_: int, depth: int = 1):
		self.__len = len_
		self.__depth = depth
		self.__states = {}

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

	def destroy(self):
		del self.__states


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
