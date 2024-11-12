import time
import typing

import torch.multiprocessing as mp

from core.utils.research.data.load.flm.file_loader import FileLoader


class Worker(mp.Process):

	def __init__(self, root_dirs: typing.List[str]):
		super().__init__()
		self.queue = mp.Queue()
		self.result_queue = mp.Queue()
		self.fileloader = FileLoader(root_dirs=root_dirs)

	def run(self):
		while True:
			idx = self.queue.get()
			self.result_queue.put((idx, self.fileloader[idx]))
