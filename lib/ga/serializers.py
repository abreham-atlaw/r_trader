from typing import *

from lib.network.rest_interface.serializers import Serializer
from lib.ga.species import Species


class PopulationSerializer(Serializer):

	def __init__(self, species_serializer: Serializer):
		super().__init__(List)
		self.__serializer = species_serializer

	def serialize(self, data: List[Species]) -> List[Dict]:
		return [
			self.__serializer.serialize(instance)
			for instance in data
		]

	def deserialize(self, json_: List[Dict]) -> List[Species]:
		return [
			self.__serializer.deserialize(instance)
			for instance in json_
		]
