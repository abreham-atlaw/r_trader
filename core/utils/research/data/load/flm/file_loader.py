import typing
from multiprocessing import Manager

import numpy as np
import torch
import os

from lib.utils.devtools import performance
from lib.utils.logger import Logger


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
			pool_size: int = 10,
			X_dir: str = "X",
			y_dir: str = "y",
			out_dtypes: typing.Type = np.float32,
			num_files: typing.Optional[int] = None,
			device=torch.device("cpu")
	):
		self.__device = device
		self.__dtype = out_dtypes
		self.root_dirs = root_dirs
		self.__X_dir = X_dir
		self.__y_dir = y_dir
		self.random_state = None
		self.__files, self.__root_dir_map = self.__get_files(size=num_files)
		self.__pool_size = pool_size
		self.manager = Manager()
		self.__init_pool()
		self.file_size = self.__get_file_size()
		# self.__pool = [None] * len(self.__files)

	@property
	def random(self):
		if self.random_state is None:
			return None
		return np.random.default_rng(self.random_state)

	def __get_file_size(self) -> int:
		first_file_name = self.__files[0]
		first_file_data = self.__load_array(os.path.join(self.root_dirs[0], self.__X_dir, first_file_name))
		return first_file_data.shape[0]

	def __init_pool(self):
		X, y = self.__load_arrays(0)

		self.__pool_X = torch.zeros(self.__pool_size, X.shape[0], X.shape[1]).to(self.__device)
		self.__pool_y = torch.zeros(self.__pool_size, y.shape[0], y.shape[1]).to(self.__device)
		self.__pool_X.share_memory_()
		self.__pool_y.share_memory_()

		self.__pool_idxs = (torch.zeros(self.__pool_size, dtype=torch.int64) - 1).share_memory_()

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
			self.__NUMPY_TORCH_TYPE_MAP[out.dtype]
		).to(self.__device).type(self.__NUMPY_TORCH_TYPE_MAP[self.__dtype])

		return tensor

	def __load_arrays(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		file_name = self.__files[idx]
		X = self.__load_array(os.path.join(self.__root_dir_map[file_name], self.__X_dir, file_name))
		y = self.__load_array(os.path.join(self.__root_dir_map[file_name], self.__y_dir, file_name))
		return X, y

	def __get_free_slot(self) -> int:

		if not torch.any(self.__pool_idxs == -1):
			self.__pool_idxs[0] = -1
			# for arr in [self.__pool_X, self.__pool_y]:
			# 	arr[0] = torch.zeros_like(arr[0])

		return list(self.__pool_idxs).index(-1)

	def __get_slot(self, idx: int) -> int:
		return self.__pool_idxs[self.__pool_idxs == idx][0]

	def __load_files(self, idx: int):

		slot = self.__get_free_slot()

		X, y = self.__load_arrays(idx)

		X.share_memory_()
		y.share_memory_()
		self.__pool_X[slot] = X
		self.__pool_y[slot] = y

		self.__pool_idxs[slot] = idx
		Logger.info(f"[+]Loaded {idx} in slot {slot}")

	def load(self, idx: int):
		if idx in self.__pool_idxs:
			return
		self.__load_files(idx)

	def __getitem__(self, item: int) -> typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]:
		if not torch.any(self.__pool_idxs == item):
			return None
		slot = self.__get_slot(item)
		return self.__pool_X[slot], self.__pool_y[slot]

	def __len__(self):
		return len(self.__files)
