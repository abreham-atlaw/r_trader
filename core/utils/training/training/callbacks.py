from typing import *

from tensorflow import keras
from tensorflow.keras.models import Model

import os
import random
from datetime import datetime

from core.utils.file_storage import DropboxClient


class Callback:

	def on_batch_end(self, core_model: Model, delta_model: Model, batch: int):
		pass

	def on_batch_start(self, core_model: Model, delta_model: Model, batch: int):
		pass

	def on_epoch_end(self, core_model: Model, delta_model: Model, epoch: int):
		pass

	def on_epoch_start(self, core_model: Model, delta_model: Model, epoch: int):
		pass


class DropboxUploadCallback(Callback):

	def __init__(
			self,
			folder="/RForexTrader",
			batch_end: bool = True,
			batch_steps: int = 1,
			epoch_end: bool = True,
	):
		self.__session_id = self.__generate_session_id()
		self.__dropbox_client = DropboxClient(folder=os.path.join(folder, "Session-%s" % (self.__session_id,)))
		self.__batch_end = batch_end
		self.__epoch_end = epoch_end
		self.__batch_steps = batch_steps

	@staticmethod
	def __generate_session_id() -> str:
		return str(random.randint(0, 100))

	@staticmethod
	def __export_model(model: keras.Model, type_: str) -> str:
		file_name = f"{type_}_{datetime.now()}.h5"
		model.save(file_name)
		return file_name

	def __save_model(self, core_model, delta_model) -> Tuple[str]:
		return tuple([
			self.__export_model(internal_model, name)
			for internal_model, name in (
				(core_model, "core"),
				(delta_model, "delta")
			)
		])

	def __upload_model(self, model_path: str):
		self.__dropbox_client.upload_file(model_path)

	def __call(self, core_model: Model, delta_model: Model):
		print("[+](Session %s)Saving Model..." % self.__session_id)
		paths = self.__save_model(core_model, delta_model)
		print("[+](Session %s)Uploading Model..." % self.__session_id)
		for path in paths:
			self.__upload_model(path)

	def on_batch_end(self, core_model: Model, delta_model: Model, batch: int):
		if self.__batch_end and (batch + 1) % self.__batch_steps == 0:
			self.__call(core_model, delta_model)

	def on_epoch_end(self, core_model: Model, delta_model: Model, epoch: int):
		if self.__epoch_end:
			self.__call(core_model, delta_model)
