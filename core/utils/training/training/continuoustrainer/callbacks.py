import typing
from abc import ABC

from tensorflow import keras

from core.utils.training.training import Trainer
from core.utils.training.training.callbacks import Callback, CheckpointUploadCallback, PCloudCheckpointUploadCallback
from .repository import TrainerRepository


class ContinuousTrainerCallback(Callback):

	def on_timeout(self, core_model: keras.Model, delta_model: keras.Model, state: Trainer.State, metrics: 'Trainer.MetricsContainer'):
		pass


class ContinuousTrainerCheckpointCallback(ContinuousTrainerCallback, CheckpointUploadCallback, ABC):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__repository, self.__id = None, None
		self.__current_checkpoint = [[None, None], None]

	def init(self, id: str, repository: TrainerRepository):
		self.__id, self.__repository = id, repository

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

	def on_timeout(self, core_model: keras.Model, delta_model: keras.Model, state: Trainer.State, metrics: 'Trainer.MetricsContainer'):
		self._call(core_model, delta_model, state)


class PCloudContinuousTrainerCheckpointCallback(PCloudCheckpointUploadCallback, ContinuousTrainerCheckpointCallback):
	pass
