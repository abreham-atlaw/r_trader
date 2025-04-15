import typing

from .staterepository import StateRepository, StateNotFoundException
from abc import ABC, abstractmethod
import os
import shutil
import pickle


class FileSystemStateRepository(StateRepository, ABC):

	def __init__(self, path: str = None, name: str = ".states", *args, **kwargs):
		super().__init__(*args, **kwargs)
		if path is None:
			path = os.path.abspath("./")

		self.__container = os.path.join(path, name)
		self.__keys = []
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
		self.__keys.append(key)

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
		self.__keys = []

	def destroy(self):
		shutil.rmtree(self.__container)
		self.__keys = []

	def get_keys(self) -> typing.List[str]:
		return self.__keys


class PickleStateRepository(FileSystemStateRepository):

	def __init__(self, file_name="content.pkl", *args, **kwargs):
		super().__init__(*args, **kwargs)
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
