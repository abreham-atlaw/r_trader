import typing

import torch
import torch.multiprocessing as mp

from core.utils.research.data.load.flm.file_loader import FileLoader
from lib.utils.logger import Logger


def load(args) -> typing.Tuple[torch.Tensor, torch.Tensor]:
	idx, *loader_args = args
	loader = FileLoader(*loader_args, pool_size=1)
	Logger.info("Loading Value")
	value = loader[idx]
	Logger.info("Returning Value")
	return value


class WorkerPool:

	def __init__(
		self,
		workers: int,
		*loader_args,
	):
		self.workers = workers
		self.__loader_args = loader_args

	def load(self, idxs: typing.List[int]) -> typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]:
		with mp.Pool(processes=self.workers) as pool:
			results = pool.map(
				load,
				[
					(idx, ) + self.__loader_args
					for idx in idxs
				]
			)
		Logger.info("Returning Results")
		return results

