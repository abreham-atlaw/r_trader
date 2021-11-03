from typing import *

import json
import cattr


class Serializer:

	def __init__(self, output_class):
		self.__output_class = output_class

	def serialize(self, data: object) -> Dict:
		return cattr.unstructure(data)

	def deserialize(self, json_: Dict) -> object:
		return cattr.structure(json_, self.__output_class)
