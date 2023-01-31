from typing import *
from abc import abstractmethod, ABC

from tensorflow import keras
from tensorflow.keras.models import Model

import os
import random
from datetime import datetime

from lib.utils.file_storage import DropboxClient, PCloudClient, FileStorage, LocalStorage
from core import Config
from .repository import MetricRepository


class Callback:

	def on_batch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		pass

	def on_batch_start(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		pass

	def on_epoch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		pass

	def on_epoch_start(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		pass


class CallbackException(Exception):
	pass


class CheckpointCallback(Callback):

	TYPES = ("core", "delta")

	def __init__(
			self,
			save_path=None,
			batch_end: bool = True,
			batch_steps: int = 1,
			epoch_end: bool = True,
			epoch_steps: int = 1
	):
		self.__batch_end = batch_end
		self.__epoch_end = epoch_end
		self.__batch_steps = batch_steps
		self.__epoch_steps = epoch_steps

		if save_path is None:
			save_path = "./"
		self.__save_path = os.path.abspath(save_path)

	def _generate_file_name(self, type_: str):
		return f"{type_}.h5"

	def _save_model(self, model: keras.Model, type_: str) -> str:
		file_path = os.path.join(self.__save_path, self._generate_file_name(type_))
		print(f"[+]Saving {type_} model to {file_path}")
		model.save(file_path)
		return file_path

	def _call(self, core_model: Model, delta_model: Model, state: 'Trainer.State'):
		for model, type_ in zip([core_model, delta_model], self.TYPES):
			file_path = self._save_model(model, type_)

	def on_batch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		if self.__batch_end and (state.batch + 1) % self.__batch_steps == 0:
			self._call(core_model, delta_model, state)

	def on_epoch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		if self.__epoch_end and (state.epoch + 1) % self.__epoch_steps == 0:
			self._call(core_model, delta_model, state)


class CheckpointUploadCallback(CheckpointCallback , ABC):

	def __init__(self, base_path: str, *args, session_id=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._session_id = session_id
		if session_id is None:
			self._session_id = self.__generate_session_id()
		self.__file_storage = self._create_filestorage(base_path)

	@staticmethod
	def __generate_session_id() -> str:
		return f"{str(random.randint(0, 10000))} - {str(datetime.now().timestamp())}"

	@abstractmethod
	def _create_filestorage(self, base_path: str) -> FileStorage:
		pass

	def _save_model(self, model: keras.Model, type_: str) -> str:
		path = super()._save_model(model, type_)
		print(f"[+]Uploading {type_} model(Session: {self._session_id})...")
		self.__file_storage.upload_file(path)
		return os.path.basename(path)


class DropboxCheckpointUploadCallback(CheckpointUploadCallback):

	def _create_filestorage(self, base_path: str) -> FileStorage:
		return DropboxClient(token=Config.DROPBOX_API_TOKEN, folder=os.path.join(base_path, "Session-%s" % (self._session_id,)))


class PCloudCheckpointUploadCallback(CheckpointUploadCallback):

	def _create_filestorage(self, base_path: str) -> FileStorage:
		return PCloudClient(token=Config.PCLOUD_API_TOKEN, folder=base_path)

	def _generate_file_name(self, type_: str):
		return f"{self._session_id}-{super()._generate_file_name(type_)}"


class LocalCheckpointUploadCallback(CheckpointUploadCallback):

	def _create_filestorage(self, base_path: str) -> FileStorage:
		return LocalStorage(base_path, port=8000)


class EarlyStoppingCallback(Callback):

	class Modes:
		MIN = -1
		MAX = 1

	class EarlyStopException(CallbackException):
		pass

	def __init__(
			self,
			model: int,
			source=1,  # Validation
			patience=0,
			value_idx=0,  # LOSS
			mode=Modes.MIN,
			verbose=False
	):
		self.__model = model
		self.__source = source
		self.__patience = patience
		self.__value_idx = value_idx
		self.__mode = mode
		if mode not in [self.Modes.MIN, self.Modes.MAX]:
			raise Exception(f"Invalid Mode: {self.__mode}")
		self.__verbose = verbose

	def __early_stop(self, metrics: 'Trainer.MetricsContainer') -> bool:
		depth = max([metric.depth for metric in metrics])
		values = [
			metric.value[self.__value_idx]
			for metric in metrics.filter_metrics(
							source=self.__source,
							depth=depth,
							model=self.__model
						)
				]
		if self.__verbose:
			print(f"EarlyStoppingCallback: Values -> {values}")
		if len(values) < self.__patience+2:
			return False
		values = values[-(self.__patience+2):]
		for i in range(self.__patience+1):
			diff = values[i+1] - values[i]
			if diff/abs(diff) == self.__mode:
				return False
		return True

	def on_epoch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		depth = max([metric.depth for metric in metrics])
		if state.depth != depth:
			return
		if self.__early_stop(metrics):
			raise EarlyStoppingCallback.EarlyStopException()


class MetricUploaderCallback(Callback):

	def __init__(self, repository: MetricRepository, on_batch_end=False, on_epoch_end=True):
		self.__repository = repository
		self.__on_batch_end = on_batch_end
		self.__on_epoch_end = on_epoch_end

	def _call(self, metrics: "Trainer.MetricsContainer"):
		for metric in metrics:
			if not self.__repository.exists(metric):
				self.__repository.write_metric(metric)

	def on_batch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		if self.__on_batch_end:
			self._call(metrics)

	def on_epoch_end(self, core_model: Model, delta_model: Model, state: 'Trainer.State', metrics: 'Trainer.MetricsContainer'):
		if self.__on_epoch_end:
			self._call(metrics)
