import json
from typing import *
from abc import ABC, abstractmethod

import pickle

from lib.network.rest_interface.serializers import Serializer


class FileIO(ABC):

	@abstractmethod
	def dumps(self, obj: Any , path: str):
		pass

	@abstractmethod
	def loads(self, path: str) -> Any:
		pass


class PickleFileIO(FileIO):

	def dumps(self, obj: Any, path: str):
		with open(path, "wb") as file:
			pickle.dump(obj, file)

	def loads(self, path: str) -> Any:
		file = open(path, "rb")
		obj = pickle.load(file)
		file.close()
		return obj


class SerializerFileIO(FileIO):

	def __init__(self, serializer: Serializer):
		self.__serializer = serializer

	def dumps(self, obj: Any, path: str):
		with open(path, "w") as file:
			print(self.__serializer.serialize_json(obj), file=file)

	def loads(self, path: str) -> Any:
		file = open(path)
		content = file.read()
		file.close()
		return self.__serializer.deserialize_json(content)


def load_json(path):
	with open(path, "r") as f:
		content = json.load(f)
	return content
