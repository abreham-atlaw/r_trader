import random
import typing

from core.utils.research.data.load.data_pool import TensorDataPool
from core.utils.research.data.load.dataloader.file_dataloader.collector import ResultCollector
from core.utils.research.data.load.dataloader.file_dataloader.worker import Worker


class WorkerManager:

	def __init__(
			self,
			workers: int,
			root_dirs: typing.List[str],
			data_pool: TensorDataPool
	):
		self.__num_workers = workers
		self.workers = [
			Worker(
				root_dirs=root_dirs
			)
			for _ in range(self.__num_workers)
		]
		self.pool = data_pool
		self.__queue = []
		self.collector = ResultCollector(
			pool=self.pool,
			workers=self.workers,
			queue=self.__queue
		)

	def is_queued(self, idx: int) -> bool:
		return idx in self.__queue

	def _select_worker(self):
		return random.choice(self.workers)

	def queue(self, idx):
		if self.is_queued(idx):
			return
		worker = self._select_worker()
		worker.queue.put(idx)
		self.__queue.append(idx)

	def start(self):
		for worker in self.workers:
			worker.start()
		# self.collector.start()
