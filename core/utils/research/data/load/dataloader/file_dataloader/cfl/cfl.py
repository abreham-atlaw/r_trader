import typing
from queue import Empty

import torch.multiprocessing as mp

from lib.utils.logger import Logger

from core.utils.research.data.load.flm.file_loader import FileLoader
from .worker_manager import WorkerManager
from ...dataloader import SpinozaDataLoader


class ConcurrentFileDataLoader(SpinozaDataLoader):

	def __init__(
			self,
			root_dirs: typing.List[str],
			workers: int,
			prefetch_factor: int = 10,
	):
		self.__fileloader = FileLoader(root_dirs=root_dirs)
		self.__num_workers = workers
		self.__root_dirs = root_dirs
		self.__prefetch_factor = prefetch_factor

	def __iter__(self):

		queue = mp.Queue()
		manager = WorkerManager(
			self.__num_workers,
			prefetch_factor=self.__prefetch_factor,
			root_dirs=self.__root_dirs,
			result_queue=queue,
		)
		manager.start()

		queues = manager.queues
		Logger.info(f"Using {len(queues)} queues")
		i = 0
		while i < len(self.__fileloader):
			for queue in queues:
				try:
					yield queue.get(timeout=0.0001)
					Logger.info(f"Yielded {i}")
					i += 1
				except Empty:

					pass

		manager.kill()

	def __len__(self):
		return len(self.__fileloader)

	def shuffle(self):
		pass
