import os.path
import typing
from datetime import datetime

import numpy as np

import math


class AgentDataGenerator:

	def __init__(
			self,
			batch_size: int,
			shape: typing.Tuple[typing.Tuple, typing.Tuple] = None,
			export_path: str = None,
			X_dir="X",
			y_dir="y"
	):
		if export_path is None:
			export_path = os.path.abspath("./")
		self.__batch_size = batch_size
		self.__X, self.__y = None, None
		if shape is not None:
			self.set_shape(shape)

		self.__export_path = os.path.abspath(export_path)
		self.__X_dir, self.__y_dir = X_dir, y_dir
		self.__save_setup = False

	@staticmethod
	def __generate_filename() -> str:
		return f"{datetime.now().timestamp()}.npy"

	@property
	def __X_path(self):
		return os.path.join(self.__export_path, self.__X_dir)

	@property
	def __y_path(self):
		return os.path.join(self.__export_path, self.__y_dir)

	def __setup_dirs(self):
		for path in [self.__X_path, self.__y_path]:
			os.mkdir(path)
		self.__save_setup = True

	def __save_array(self, arr: np.ndarray, path: str):
		np.save(path, arr)

	def __get_idxs(self, idx: int) -> typing.Tuple[int, int]:
		return idx*self.__batch_size, (idx+1)*self.__batch_size

	def set_shape(self, shape: typing.Tuple[typing.Tuple, typing.Tuple]):
		self.__X, self.__y = np.zeros((0, *shape[0])), np.zeros((0, *shape[1]))

	def concatenate(self, X: np.ndarray, y: np.ndarray):
		if self.__X is None:
			self.set_shape((X.shape[1:], y.shape[1:]))
		self.__X = np.concatenate((self.__X, X))
		self.__y = np.concatenate((self.__y, y))

	def append(self, X: np.ndarray, y: np.ndarray):
		self.concatenate(np.expand_dims(X, 0), np.expand_dims(y, 0))

	def remove(self, idx):
		si, ei = self.__get_idxs(idx)
		self.__X = np.concatenate((self.__X[:si], self.__X[ei:]))
		self.__y = np.concatenate((self.__y[:si], self.__y[ei:]))

	def save(self, filename: str = None):
		if filename is None:
			filename = self.__generate_filename()
		if not self.__save_setup:
			self.__setup_dirs()
		for arr, path in zip([self.__X, self.__y], [self.__X_path, self.__y_path]):
			self.__save_array(
				arr,
				os.path.join(path, filename)
			)

	def __getitem__(self, idx: int):
		si, ei = self.__get_idxs(idx)
		return self.__X[si: ei], self.__y[si: ei]

	def __len__(self):
		return math.ceil((self.__X.shape[0]+1) / self.__batch_size)
