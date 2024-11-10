import typing

import torch
import numpy as np

from lib.utils.decorators.thread_decorator import thread_method
from lib.utils.worker_manager import WorkerManager
from ..spinoza_dataloader import SpinozaDataLoader
from ...data_pool import DataPool
from ...flm.file_loader import FileLoader
from .worker_pool import WorkerPool

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
		self.__pool = DataPool(pool_size)
		self.__loader = FileLoader(pool=self.__pool, root_dirs=root_dirs, X_dir=X_dir, y_dir=y_dir)
		self.__num_workers = workers
		self.__preload_size = preload_size
		self.__worker_pool = WorkerPool(
			workers,
			root_dirs,
		)

	@property
	def has_preload(self) -> bool:
		return self.__num_workers > 0 and self.__preload_size > 0

	@thread_method
	def __preload(self, idx: int):

		idxs = [
			i
			for i in range(idx, idx + self.__preload_size)
			if i not in self.__pool
		]

		results = self.__worker_pool.load(idxs)
		for idx, result in zip(idxs, results):
			self.__pool[idx] = result

	def shuffle(self):
		self.__loader.shuffle()

	def __len__(self) -> int:
		return len(self.__loader)

	def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		value = self.__loader[idx]
		if self.has_preload:
			self.__preload(idx + 100)

		return value
