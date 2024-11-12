from queue import Empty
from threading import Thread

import torch.multiprocessing as mp

import math
import typing

from core.utils.research.data.load.flm.file_loader import FileLoader
from .worker import Worker


class WorkerManager:

	def __init__(self, workers: int, prefetch_factor: int, root_dirs: typing.List[str], result_queue: mp.Queue):
		super().__init__()
		self.__num_workers = workers
		self.__prefetch_factor = prefetch_factor
		self.__root_dirs = root_dirs
		self.__workers = []
		self.__queue = result_queue
		self.__fileloader = FileLoader(root_dirs=root_dirs, use_pool=False)
		self.__loading_queue = mp.Queue()
		self.__result_queues = []
		self.__kill_switch = False

	@property
	def queues(self) -> typing.List[mp.Queue]:
		return [
			worker.result_queue
			for worker in self.__workers
		]

	def __generate_ranges(self) -> typing.List[typing.Tuple[int, int]]:
		return [
			(i*self.__prefetch_factor, min(len(self.__fileloader), (i+1)*self.__prefetch_factor))
			for i in range(math.ceil(len(self.__fileloader) / self.__prefetch_factor))
		]

	def __init_worker(self):
		worker = Worker(
			root_dirs=self.__root_dirs,
			queue=self.__queue,
			loading_queue=self.__loading_queue
		)
		worker.start()
		return worker

	def __init_workers(self):
		for range_ in self.__generate_ranges():
			self.__loading_queue.put(range_)
		self.__workers = [
			self.__init_worker()
			for _ in range(self.__num_workers)
		]

	def __sync_workers(self):

		for i in range(self.__num_workers):
			if self.__workers[i].is_alive():
				continue
			self.__workers[i] = self.__init_worker()

	def __sync_queues(self):
		for worker in self.__workers:
			try:
				self.__queue.put(
					worker.result_queue.get(timeout=0.001)
				)
			except Empty:
				pass

	def start(self):
		self.__init_workers()

	def kill(self):
		for worker in self.__workers:
			worker.kill()
		self.__kill_switch = True
