import time
import typing

import torch

from lib.utils.logger import Logger
from .file_loader import FileLoader
from .file_loader_thread import FileLoaderThread


class FileLoadManager:

	def __init__(
			self,
			file_loader: FileLoader,
			preload_size: int = 5
	):

		self.__preload_size = preload_size
		self.fileloader = file_loader
		self.__fileloader_thread = FileLoaderThread(self.fileloader)

	def __wait_and_get(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		data = self.fileloader[idx]
		if data is not None:
			return data
		self.__fileloader_thread.queue(idx, urgent=True)
		Logger.info(f"Waiting for file {idx}...")
		time.sleep(0.1)
		return self.__wait_and_get(idx)

	def __preload(self, idx: int):
		for i in range(idx, idx+self.__preload_size):
			if i >= len(self.fileloader):
				break
			self.__fileloader_thread.queue(i)

	def __getitem__(self, item: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		data = self.__wait_and_get(item)
		self.__preload(item + 1)
		return data

	def start(self):
		Logger.info(f"Starting File Load Thread...")
		self.__fileloader_thread.start()
		self.__fileloader_thread.queue(0)

	def stop(self):
		Logger.info(f"Stopping File Load Thread...")
		self.__fileloader_thread.kill()
