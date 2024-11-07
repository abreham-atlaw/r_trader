import numpy as np
import torch
import typing
from torch.utils.data import Dataset

import random
from collections import OrderedDict
import threading
import os

from lib.utils.decorators.thread_decorator import thread_method


class BaseDataset(Dataset):

	__NUMPY_TORCH_TYPE_MAP = {
		np.dtype('int8'): torch.int8,
		np.dtype('int32'): torch.int32,
		np.dtype('int64'): torch.int64,
		np.dtype('uint8'): torch.uint8,
		np.dtype('float16'): torch.float16,
		np.dtype('float32'): torch.float32,
		np.dtype('float64'): torch.float64,
	}

	def __init__(
			self,
			root_dirs: list,
			cache_size: int = 5,
			X_dir: str = "X",
			y_dir: str = "y",
			out_dtypes: typing.Type = np.float32,
			num_files: typing.Optional[int] = None,
			preload: bool = False,
			preload_size = 3
	):
		self.__dtype = out_dtypes
		self.root_dirs = root_dirs
		self.__X_dir = X_dir
		self.__y_dir = y_dir
		self.random_state = None

		self.__files, self.__root_dir_map = self.__get_files(size=num_files)

		self.cache = OrderedDict()
		self.cache_size = cache_size
		self.data_points_per_file = self.__get_dp_per_file()

		self.__preload = preload
		self.__preload_size = preload_size

	@property
	def random(self):
		if self.random_state is None:
			return None
		return np.random.default_rng(self.random_state)

	def shuffle(self):
		print("[+]Shuffling dataset...")
		random.shuffle(self.__files)
		self.cache = OrderedDict()
		self.random_state = random.randint(0, 1000)

	def __get_dp_per_file(self) -> int:
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

	def __len__(self):
		return len(self.__files) * self.data_points_per_file

	def __preload_file(self, idx: str):
		filename = self.__files[idx]
		root_dir = self.__root_dir_map[filename]

		X, y = [
			self.__load_array(os.path.join(root_dir, dir_, filename))
			for dir_ in [self.__X_dir, self.__y_dir]
		]
		self.cache[idx] = (X, y)

	@thread_method
	def __preload_files(self, idx: str):
		for i in range(idx, idx+self.__preload_size):
			if i in self.cache or i >= len(self.__files):
				continue
			self.__preload_file(i)

	def __load_array(self, path: str) -> np.ndarray:
		out = np.load(path).astype(self.__dtype)
		indexes = np.arange(out.shape[0])
		if self.random is not None:
			self.random.shuffle(indexes)

		return out[indexes]

	def __load_files(self, idx: np.ndarray) -> np.ndarray:
		if idx in self.cache:
			return self.cache[idx]

		if len(self.cache) >= self.cache_size:
			self.cache.popitem(last=False)

		file_name = self.__files[idx]
		root_dir = self.__root_dir_map[file_name]

		X = self.__load_array(os.path.join(root_dir, self.__X_dir, file_name))
		y = self.__load_array(os.path.join(root_dir, self.__y_dir, file_name))

		self.cache[idx] = (X, y)

		if self.__preload:
			self.__preload_files(idx+1)

		return X, y

	def __getitem__(self, idx):
		file_idx = idx // self.data_points_per_file
		data_idx = idx % self.data_points_per_file

		X, y = self.__load_files(file_idx)

		return tuple([torch.from_numpy(dp[data_idx]).type(self.__NUMPY_TORCH_TYPE_MAP[dp.dtype]) for dp in [X, y]])
