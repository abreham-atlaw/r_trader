import time
from threading import Thread

import torch

from .file_loader import FileLoader


class FileLoaderThread(Thread):

	def __init__(
			self,
			file_loader: FileLoader,
	):
		super().__init__()
		self.__file_loader = file_loader
		self.__queue = self.__file_loader.manager.list()
		self.__kill = self.__file_loader.manager.Value('i', False)

	def queue(self, idx, urgent=False):
		if idx in self.__queue:
			return
		if urgent:
			self.__queue.insert(0, idx)
		else:
			self.__queue.append(idx)

	def run(self):
		while True:
			if self.__kill.value:
				break
			if len(self.__queue) == 0:
				time.sleep(0.01)
				continue
			self.__file_loader.load(self.__queue.pop(0))

	def kill(self):
		self.__kill.value = True
