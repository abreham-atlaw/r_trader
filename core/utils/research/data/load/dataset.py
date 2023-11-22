import random
from collections import OrderedDict

import numpy as np
import torch
import typing
from torch.utils.data import Dataset

import os


class BaseDataset(Dataset):
	def __init__(
			self,
			root_dirs: list,
			cache_size: int = 5,
			X_dir: str = "X",
			y_dir: str = "y",
			out_dtypes: typing.Type = np.float32,
	):
		self.__dtype = out_dtypes
		self.root_dirs = root_dirs
		self.__X_dir = X_dir
		self.__y_dir = y_dir

		self.__files, self.__root_dir_map = self.__get_files()
		self.cache = OrderedDict()
		self.cache_size = cache_size
		self.data_points_per_file = self.__get_dp_per_file()

	def shuffle(self):
		random.shuffle(self.__files)
		self.cache = OrderedDict()

	def __get_dp_per_file(self) -> int:
		first_file_name = self.__files[0]
		first_file_data = self.__load_array(os.path.join(self.root_dirs[0], self.__X_dir, first_file_name))
		return first_file_data.shape[0]

	def __get_files(self):
		files_map = {}
		files = []
		for root_dir in self.root_dirs:
			X_encoder_path = os.path.join(root_dir, self.__X_dir)

			dir_files = sorted(os.listdir(X_encoder_path))
			files += dir_files
			for file in dir_files:
				files_map[file] = root_dir
		return files, files_map

	def __len__(self):
		return len(self.__files) * self.data_points_per_file

	def __load_array(self, path: str) -> np.ndarray:
		return np.load(path).astype(self.__dtype)

	def __getitem__(self, idx):
		file_idx = idx // self.data_points_per_file
		data_idx = idx % self.data_points_per_file

		if file_idx not in self.cache:
			if len(self.cache) >= self.cache_size:
				self.cache.popitem(last=False)

			file_name = self.__files[file_idx]
			root_dir = self.__root_dir_map[file_name]

			X = self.__load_array(os.path.join(root_dir, self.__X_dir, file_name))
			y = self.__load_array(os.path.join(root_dir, self.__y_dir, file_name))

			self.cache[file_idx] = (X, y)

		return tuple([torch.from_numpy(dp[data_idx]) for dp in self.cache[file_idx]])