from typing import *

import os

from lib.network.rest_interface.serializers import Serializer
from lib.utils.fileio import PickleFileIO, SerializerFileIO
from .species import Species
from .serializers import PopulationSerializer


class Callback:

	def __init__(self):
		pass

	def on_epoch_end(self, population: List[Species]):
		pass

	def on_epoch_start(self, population: List[Species]):
		pass


class CheckpointCallback(Callback):

	def __init__(self, species_serializer: Optional[Serializer] = None, save_path: str = None):
		super().__init__()

		self.__fileio = PickleFileIO()
		if species_serializer is not None:
			self.__fileio = SerializerFileIO(
				PopulationSerializer(
					species_serializer
				)
			)
		self.__save_path = save_path
		if save_path is None:
			self.__save_path = os.path.abspath("population.ga")

	def on_epoch_end(self, population: List[Species]):
		self.__fileio.dumps(population, self.__save_path)
