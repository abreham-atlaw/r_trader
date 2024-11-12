import typing

import torch.multiprocessing as mp

from core.utils.research.data.load.flm.file_loader import FileLoader
from lib.utils.logger import Logger


class Worker(mp.Process):

	def __init__(self, queue: mp.Queue, loading_queue: mp.Queue,root_dirs: typing.List[str], range: typing.Tuple[int, int] = None):
		super().__init__()
		self.result_queue = mp.Queue()
		self.loading_queue = loading_queue
		self.root_dirs = root_dirs
		self.range = range

	def run(self):
		fileloader = FileLoader(
			root_dirs=self.root_dirs,
			use_pool=False
		)

		while True:
			range_ = self.loading_queue.get()
			for i in range(*range_):
				self.result_queue.put(fileloader[i])
			Logger.info(f"Completed {range_}")
