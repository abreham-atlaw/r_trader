import typing
from multiprocessing import Manager
import random

import numpy as np
import torch
import os

from core.utils.research.data.load.data_pool import DataPool


class FileLoader:

	__NUMPY_TORCH_TYPE_MAP = {
		np.dtype('int8'): torch.int8,
		np.dtype('int32'): torch.int32,
		np.dtype('int64'): torch.int64,
		np.dtype('uint8'): torch.uint8,
		np.dtype('float16'): torch.float16,
		np.dtype('float32'): torch.float32,
		np.float32: torch.float32,
		np.dtype('float64'): torch.float64,
	}

	def __init__(
			self,
			root_dirs: list,
			pool: DataPool = None,
			pool_size: int = 5,
			X_dir: str = "X",
			y_dir: str = "y",
			out_dtypes: typing.Type = np.float32,
			num_files: typing.Optional[int] = None,
			random_state=None
	):
		self.__dtype = out_dtypes
		self.root_dirs = root_dirs
		self.__X_dir = X_dir
		self.__y_dir = y_dir
		self.random_state = random_state
		self.__files, self.__root_dir_map = self.__get_files(size=num_files)
		self.__pool_size = pool_size

		if pool is None:
			pool = DataPool(pool_size)

		self.__pool = pool
		self.file_size = self.__get_file_size()

	@property
	def random(self):
		if self.random_state is None:
			return None
		return np.random.default_rng(self.random_state)

	def __get_file_size(self) -> int:
		first_file_name = self.__files[0]
		first_file_data = self.__load_array(os.path.join(self.root_dirs[0], self.__X_dir, first_file_name))
		return first_file_data.shape[0]

	def __get_files(self, size=None):
		files_map = {}
		files = []
		for root_dir in self.root_dirs:
			if size is not None and len(files) >= size:
				break
			X_encoder_path = os.path.join(root_dir, self.__X_dir)
			dir_files = sorted(os.listdir(X_encoder_path))
			if size is not None:
				dir_files = dir_files[:size - len(files)]
			files += dir_files
			for file in dir_files:
				files_map[file] = root_dir
		return files, files_map

	def __load_array(self, path: str) -> torch.Tensor:
		out: np.ndarray = np.load(path).astype(self.__dtype)
		indexes = np.arange(out.shape[0])
		if self.random is not None:
			self.random.shuffle(indexes)

		tensor = torch.from_numpy(out[indexes]).type(
			self.__NUMPY_TORCH_TYPE_MAP[self.__dtype]
		)

		return tensor

	def __load_arrays(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		file_name = self.__files[idx]
		X = self.__load_array(os.path.join(self.__root_dir_map[file_name], self.__X_dir, file_name))
		y = self.__load_array(os.path.join(self.__root_dir_map[file_name], self.__y_dir, file_name))
		return X, y

	def __load_files(self, idx: int):
		cached = self.__pool[idx]
		if cached is not None:
			return cached

		X, y = self.__load_arrays(idx)

		self.__pool[idx] = (X, y)

		return X, y

	def load(self, idx: int):
		return self.__load_files(idx)

	def shuffle(self):
		print("[+]Shuffling dataset...")
		random.shuffle(self.__files)
		self.random_state = random.randint(0, 1000)

	def __getitem__(self, item: int) -> typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]:
		return self.load(item)

	def __len__(self):
		return len(self.__files)
