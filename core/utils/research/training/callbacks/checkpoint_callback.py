import os.path
from datetime import datetime

import torch

from core.utils.research.training.callbacks import Callback
from lib.utils.file_storage import FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class CheckpointCallback(Callback):
	def __init__(self, path):
		self.path = path

	@staticmethod
	def __generate_name() -> str:
		return f"{datetime.now().timestamp()}.pt"

	def on_epoch_end(self, model, epoch, logs=None):
		path = self.path
		if os.path.isdir(self.path):
			path = os.path.join(self.path, self.__generate_name())
		ModelHandler.save(model, path)
		return path


class StoreCheckpointCallback(CheckpointCallback):

	def __init__(self, fs: FileStorage, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__file_storage = fs

	def on_epoch_end(self, model, epoch, logs=None):
		path = super().on_epoch_end(model, epoch, logs)
		self.__file_storage.upload_file(path)
