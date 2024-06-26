import os.path
from datetime import datetime

import torch

from core.utils.research.training.callbacks import Callback
from lib.utils.file_storage import FileStorage
from lib.utils.torch_utils.model_handler import ModelHandler


class CheckpointCallback(Callback):
	def __init__(self, path, save_state=False):
		self.path = path
		self.save_state = save_state

	def _generate_name(self) -> str:
		ext = "pt"
		if self.save_state:
			ext = "pth"
		return f"{datetime.now().timestamp()}.{ext}"

	def _save(self, model, path):
		if self.save_state:
			torch.save(model.state_dict(), path)
		else:
			ModelHandler.save(model, path)

	def on_epoch_end(self, model, epoch, losses, logs=None):
		path = self.path
		if os.path.isdir(self.path):
			path = os.path.join(self.path, self._generate_name())
		self._save(model, path)
		return path


class StoreCheckpointCallback(CheckpointCallback):

	def __init__(self, fs: FileStorage, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__file_storage = fs

	def on_epoch_end(self, model, epoch, loss, logs=None):
		path = super().on_epoch_end(model, epoch, logs)
		self.__file_storage.upload_file(path)
