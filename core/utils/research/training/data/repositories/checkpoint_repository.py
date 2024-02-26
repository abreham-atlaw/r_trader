import os
import typing

from torch import nn

from core import Config
from core.utils.research.training.data.repositories.state_repository import TrainingStateRepository
from core.utils.research.training.data.state import TrainingState
from lib.utils.file_storage import PCloudClient, FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class CheckpointRepository:

	def __init__(self, file_storage: FileStorage, tmp_path:str = "./"):
		self.__state_repository = TrainingStateRepository()
		if file_storage is None:
			file_storage = PCloudClient(
				Config.PCLOUD_API_TOKEN,
				Config.PCLOUD_FOLDER
			)
		self.__file_storage = file_storage
		self.__tmp_path = tmp_path

	def __generate_filename(self, id: str) -> str:
		return f"{id}.zip"

	def __save_model(self, model: nn.Module, state: TrainingState):
		file_path = self.__generate_filename(state.id)
		ModelHandler.save(model, file_path)
		self.__file_storage.upload_file(file_path)

	def update(self, state: TrainingState, model: nn.Module):
		self.__state_repository.save(state)
		self.__save_model(model, state)

	def get(self, id: str) -> typing.Optional[typing.Tuple[TrainingState, nn.Module]]:
		state = self.__state_repository.get(id)
		if state is None:
			return None
		filepath = self.__generate_filename(id)
		download_path = os.path.join(self.__tmp_path, filepath)
		self.__file_storage.download(filepath, download_path=download_path)
		return state, ModelHandler.load(filepath)
