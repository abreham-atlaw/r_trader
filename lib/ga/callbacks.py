from typing import *

import os
from datetime import datetime

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

	def _get_save_path(self) -> str:
		if self._save_path is not None:
			return self._save_path
		return os.path.abspath(f"{datetime.now().timestamp()}.ga")

	def on_epoch_end(self, population: List[Species]):
		path = self._get_save_path()
		self.__fileio.dumps(population, path)
		return path


class StoreCheckpointCallback(CheckpointCallback):

	def __init__(self, fs: FileStorage, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__fs = fs

	def on_epoch_end(self, population: List[Species]):
		path = super().on_epoch_end(population)
		self.__fs.upload_file(path)
