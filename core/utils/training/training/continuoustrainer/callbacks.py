import typing
from abc import ABC

from tensorflow import keras

import os
import json

from core.utils.training.training import Trainer
from core.utils.training.training.callbacks import Callback, CheckpointUploadCallback, PCloudCheckpointUploadCallback
from .repository import TrainerRepository


class ContinuousTrainerCallback(Callback):

	def on_timeout(
			self,
			core_model: keras.Model,
			delta_model: keras.Model,
			state: Trainer.State,
			metrics: 'Trainer.MetricsContainer'
	):
		pass


class ContinuousTrainerCheckpointCallback(ContinuousTrainerCallback, CheckpointUploadCallback, ABC):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__repository, self.__id = None, None
		self.__current_checkpoint = [[None, None], None]

	def init(self, id_: str, repository: TrainerRepository):
		self.__id, self.__repository = id_, repository

	def __get_repository(self) -> TrainerRepository:
		if self.__repository is None:
			raise ValueError("Not Initiated")
		return self.__repository

	def __get_id(self) -> str:
		if self.__id is None:
			raise ValueError("Not Initiated")
		return self.__id

	def _save_model(self, model: keras.Model, type_: str) -> str:
		path = super()._save_model(model, type_)
		self.__current_checkpoint[0][self.TYPES.index(type_)] = path
		return path

	def _call(self, core_model: keras.Model, delta_model: keras.Model, state: 'Trainer.State'):
		self.__current_checkpoint = [[None, None], state]
		super()._call(core_model, delta_model, state)
		if None not in self.__current_checkpoint[0]:
			self.__get_repository().update_checkpoint(self.__get_id(), *self.__current_checkpoint)

	def on_timeout(
			self,
			core_model: keras.Model,
			delta_model: keras.Model,
			state: Trainer.State,
			metrics: 'Trainer.MetricsContainer'
	):
		self._call(core_model, delta_model, state)


class PCloudContinuousTrainerCheckpointCallback(PCloudCheckpointUploadCallback, ContinuousTrainerCheckpointCallback):
	pass


class RecursiveNotebookCallback(ContinuousTrainerCallback):

	def __init__(
			self,
			username: str,
			key: str,
			kernel: str,
			meta_data: typing.Dict = None,
			notebook_pull_path: str = None
	):
		super().__init__()
		self.__api = self.__create_api(username, key)
		self.__kernel = kernel
		self.__meta_data = meta_data
		if meta_data is None:
			self.__meta_data = {}
		self.__pull_path = notebook_pull_path
		if notebook_pull_path is None:
			self.__pull_path = f"./notebook-{self.__kernel}"

	@staticmethod
	def __create_api(username, key):
		os.environ["KAGGLE_USERNAME"] = username
		os.environ["KAGGLE_KEY"] = key
		from kaggle.api.kaggle_api_extended import KaggleApi
		api = KaggleApi()
		api.authenticate()
		return api

	@staticmethod
	def __update_meta(meta_data, path):
		if len(meta_data) == 0:
			return

		meta_path = os.path.join(path, "kernel-metadata.json")
		with open(meta_path, "r") as file:
			meta = json.load(file)

		meta.update(meta_data)

		with open(meta_path, "w") as file:
			json.dump(meta, file)

	def __pull_notebook(self, kernel: str, pull_path: str):
		os.mkdir(pull_path)
		self.__api.kernels_pull(kernel, pull_path, metadata=True)

	def __push_notebook(self, path):
		self.__api.kernels_push(path)

	def on_timeout(self, core_model: keras.Model, delta_model: keras.Model, state: Trainer.State, metrics: Trainer.MetricsContainer):
		self.__pull_notebook(self.__kernel, self.__pull_path)
		self.__update_meta(self.__meta_data, self.__pull_path)
		self.__push_notebook(self.__pull_path)

