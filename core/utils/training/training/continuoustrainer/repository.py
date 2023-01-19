import typing
from abc import ABC, abstractmethod

import os

from lib.utils.file_storage import FileStorage, PCloudClient
from core.utils.training.training import Trainer
from core import Config


class TrainerRepository(ABC):

	@abstractmethod
	def update_checkpoint(self, id: str, paths: typing.Tuple[str, str], epoch: int):  # TODO: epoch: int  => state: Trainer.State
		pass

	@abstractmethod
	def get_checkpoint(self, id: str) -> typing.Optional[typing.Tuple[typing.Tuple[str, str], int]]:
		pass


class FileStorageTrainerRepository(TrainerRepository, ABC):

	__DELIMITER = "\n"

	def __init__(self, base_path: str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__file_storage = self._create_file_storage(base_path)

	@abstractmethod
	def _create_file_storage(self, base_path: str) -> FileStorage:
		pass

	def update_checkpoint(self, id: str, paths: typing.Tuple[str, str], epoch: int):
		file = open(id, "w")
		print(f"{paths[0]}{self.__DELIMITER}{paths[1]}{self.__DELIMITER}{epoch}", file=file, end="")
		file.close()
		self.__file_storage.upload_file(id)

	def get_checkpoint(self, id: str) -> typing.Optional[typing.Tuple[typing.Tuple[str, str], int]]:
		try:
			checkpoint_url = self.__file_storage.get_url(id)
		except Exception:
			return None

		os.system(f"wget {checkpoint_url} -O {id}")
		with open(id) as file:
			core_url, delta_url, epoch = file.read().split(self.__DELIMITER)
		return (core_url, delta_url), int(epoch)


class PCloudTrainerRepository(FileStorageTrainerRepository):

	def _create_file_storage(self, base_path: str) -> FileStorage:
		return PCloudClient(Config.PCLOUD_API_TOKEN, folder=base_path)
