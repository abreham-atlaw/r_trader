from typing import *

import cattr
import json

import numpy as np


class Serializer:

	def __init__(self, output_class):
		self.__output_class = output_class

	def serialize(self, data: object) -> Dict:
		return cattr.unstructure(data)

	def serialize_json(self, data: object):
		return json.dumps(self.serialize(data))

	def deserialize(self, json_: Dict) -> object:
		return cattr.structure(json_, self.__output_class)

	def deserialize_json(self, json_: str):
		return self.deserialize(json.loads(json_))


class NumpySerializer(Serializer):

	def __init__(self):
		super().__init__(np.ndarray)

	def serialize(self, data: np.ndarray) -> object:
		return data.tolist()

	def deserialize(self, json_: List) -> np.ndarray:
		return np.array(json_)
