import typing

import torch
import numpy as np

from lib.utils.decorators.thread_decorator import thread_method
from lib.utils.devtools import performance
from lib.utils.logger import Logger
from .collector import ResultCollector
from .worker import Worker
from ..spinoza_dataloader import SpinozaDataLoader
from ...data_pool import TensorDataPool, MapDataPool
from ...flm.file_loader import FileLoader
from .worker_pool import WorkerPool
from .worker_manager import WorkerManager


class FileDataLoader(SpinozaDataLoader):

	def __init__(
			self,
			root_dirs: typing.List[str],
			X_dir: str = "X",
			y_dir: str = "y",
			pool_size: int = 5,
			workers: int = 0,
			preload_size: int = 3,
			out_dtypes: typing.Type = np.float32
	):
		self.__dtype = out_dtypes
		self.__pool = MapDataPool(pool_size)
		self.__loader = FileLoader(pool=self.__pool, root_dirs=root_dirs, X_dir=X_dir, y_dir=y_dir)
		self.__num_workers = workers
		self.__preload_size = preload_size
		if self.__num_workers > 0:
			self.__init_workers()

	def __init_workers(self):
		self.__worker_manager = WorkerManager(
			workers=self.__num_workers,
			root_dirs=self.__loader.root_dirs,
			data_pool=self.__pool
		)
		self.__worker_manager.start()

	@property
	def has_preload(self) -> bool:
		return self.__num_workers > 0 and self.__preload_size > 0

	# @performance.track_func_performance()
	@thread_method
	def __preload(self, idx: int):
		for i in range(idx, idx+self.__preload_size):
			self.__worker_manager.queue(i)

	def shuffle(self):
		self.__loader.shuffle()

	def __len__(self) -> int:
		return len(self.__loader)

	def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		value = self.__loader[idx]
		if self.has_preload:
			self.__preload(idx + 50)

		return value
