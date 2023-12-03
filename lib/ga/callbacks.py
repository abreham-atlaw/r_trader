from typing import *

import os

from lib.network.rest_interface.serializers import Serializer
from lib.utils.fileio import PickleFileIO, SerializerFileIO
from .species import Species
from .serializers import PopulationSerializer
from ..utils.file_storage import FileStorage


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
		self._save_path = save_path
		if save_path is None:
			self._save_path = os.path.abspath("population.ga")

	def on_epoch_end(self, population: List[Species]):
		self.__fileio.dumps(population, self._save_path)


class StoreCheckpointCallback(CheckpointCallback):

	def __init__(self, fs: FileStorage, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__fs = fs

	def on_epoch_end(self, population: List[Species]):
		super().on_epoch_end(population)
		self.__fs.upload_file(self._save_path)
