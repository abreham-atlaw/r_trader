import os.path
import random
from datetime import datetime

import torch

from core.Config import MODEL_SAVE_EXTENSION
from core.di import ServiceProvider
from core.utils.research.training.callbacks import Callback
from lib.utils.file_storage import FileStorage
from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler


class CheckpointCallback(Callback):
	def __init__(
			self,
			path,
			save_state=False,
			ext=MODEL_SAVE_EXTENSION,
			interval: int = None,
			simplified_mode: bool = False
	):
		self.path = path
		self.save_state = save_state
		self.ext = ext
		self.interval = interval
		self.simplified_mode = simplified_mode

	def _generate_name(self) -> str:
		return f"{datetime.now().timestamp()}.{self.ext}"

	def _save(self, model, path):
		ModelHandler.save(model, path)

	def on_epoch_end(self, model, epoch, losses, logs=None):
		if self.simplified_mode and hasattr(model, "model"):
			Logger.info(f"Using internal model...")
			model = model.model
		if self.interval is not None and epoch % self.interval != 0:
			return
		path = self.path
		if os.path.isdir(self.path):
			path = os.path.join(self.path, self._generate_name())
		self._save(model, path)
		return path


class StoreCheckpointCallback(CheckpointCallback):

	def __init__(self, *args, fs: FileStorage = None, delete_stored=False, active=False, **kwargs):
		super().__init__(*args, **kwargs)
		if fs is None:
			fs = ServiceProvider.provide_file_storage()
		self.__file_storage = fs
		self.__delete_stored = delete_stored
		self.__active = active

	def on_epoch_end(self, model, epoch, loss, logs=None):
		if not self.__active:
			return
		path = super().on_epoch_end(model, epoch, logs)
		if path is None:
			return
		self.__file_storage.upload_file(path)
		if self.__delete_stored:
			os.remove(path)


class RandomCheckpointStoreCallback(StoreCheckpointCallback):

	def __init__(self, probability: float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__probability = probability

	def on_epoch_end(self, model, epoch, loss, logs=None):
		if random.random() < self.__probability:
			return super().on_epoch_end(model, epoch, loss, logs)
		return


class TrainEndCheckpointStoreCallback(StoreCheckpointCallback):

	def on_train_end(self, model, logs=None):
		super().on_epoch_end(model, 0, logs)

	def on_epoch_end(self, model, epoch, loss, logs=None):
		pass
