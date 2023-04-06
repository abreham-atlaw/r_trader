import typing
from abc import ABC, abstractmethod

import os
import json

from lib.utils.file_storage import FileStorage, PCloudClient, LocalStorage, FileNotFoundException
from core.utils.training.training import Trainer
from core import Config


class TrainerRepository(ABC):

	@abstractmethod
	def update_checkpoint(self, id: str, paths: typing.Tuple[str, str], state: Trainer.State):  # TODO: epoch: int  => state: Trainer.State
		pass

	@abstractmethod
	def get_checkpoint(self, id: str) -> typing.Optional[typing.Tuple[typing.Tuple[str, str], Trainer.State]]:
		pass


class FileStorageTrainerRepository(TrainerRepository, ABC):

	PATH_KEY = "path-{}"

	def __init__(self, base_path: str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__file_storage = self._create_file_storage(base_path)

	@abstractmethod
	def _create_file_storage(self, base_path: str) -> FileStorage:
		pass
	@staticmethod
	def __serialize_state(state: Trainer.State) -> typing.Dict:
		return state.__dict__.copy()

	@staticmethod
	def __deserialize_state(json_: typing.Dict) -> Trainer.State:
		state = Trainer.State(None, None, None, None)
		state.__dict__ = json_.copy()
		return state

	def update_checkpoint(self, id: str, paths: typing.Tuple[str, str], state: Trainer.State):
		file = open(id, "w")

		json_ = self.__serialize_state(state)
		json_.update({
			FileStorageTrainerRepository.PATH_KEY.format(i): path
			for i, path in enumerate(paths)
		})

		json.dump(json_, file)
		file.close()
		self.__file_storage.upload_file(id)

	def get_checkpoint(self, id: str) -> typing.Optional[typing.Tuple[typing.Tuple[str, str], Trainer.State]]:
		try:
			checkpoint_url = self.__file_storage.get_url(id)
		except FileNotFoundException:
			return None

		os.system(f"wget {checkpoint_url} -O {id}")
		with open(id, "r") as file:
			json_: typing.Dict = json.load(file)
		paths: typing.Tuple[str, str] = tuple([json_.pop(FileStorageTrainerRepository.PATH_KEY.format(i)) for i in range(2)])
		state = self.__deserialize_state(json_)
		return paths, state


class PCloudTrainerRepository(FileStorageTrainerRepository):

	def _create_file_storage(self, base_path: str) -> FileStorage:
		return PCloudClient(Config.PCLOUD_API_TOKEN, folder=base_path)


class LocalFileStorageRepository(FileStorageTrainerRepository):

	def _create_file_storage(self, base_path: str) -> FileStorage:
		return LocalStorage(base_path, port=8000)
