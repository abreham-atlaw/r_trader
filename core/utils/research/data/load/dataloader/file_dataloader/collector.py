import time
import typing
from threading import Thread

from core.utils.research.data.load.data_pool import TensorDataPool
from core.utils.research.data.load.dataloader.file_dataloader.worker import Worker


class WorkerResultCollector(Thread):

	def __init__(
			self,
			pool: TensorDataPool,
			worker: Worker,
			queue: typing.List[int],
	):
		super().__init__()
		self.__pool = pool
		self.__worker = worker
		self.__queue = queue

	def run(self):
		while True:
			idx, result = self.__worker.result_queue.get()
			self.__pool[idx] = result
			self.__queue.remove(idx)


class ResultCollector:

	def __init__(
			self,
			pool: TensorDataPool,
			workers: typing.List[Worker],
			queue: typing.List[int],
	):
		super().__init__()
		self.collectors = [
			WorkerResultCollector(
				pool=pool,
				worker=worker,
				queue=queue
			)
			for worker in workers
		]

	def start(self):
		for collector in self.collectors:
			collector.start()
