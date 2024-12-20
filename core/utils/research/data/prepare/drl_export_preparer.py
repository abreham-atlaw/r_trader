import numpy as np

import os
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split

from core.utils.research.model.layers import MovingAverage


class DRLExportPreparer:

	def __init__(
			self,
			input_size: int,
			seq_len: int,
			ma_window: int,
			export_path: str,

			train_dir_name: str = "train",
			test_dir_name: str = "test",
			X_dir_name: str = "X",
			y_dir_name: str = "y",

			test_split_size: float = 0.2,

			split_shuffle: bool = True

	):
		self.__input_size = input_size
		self.__seq_len = seq_len
		self.__ma_window = ma_window
		self.__export_path = export_path

		self.__train_dir_name, self.__test_dir_name = train_dir_name, test_dir_name
		self.__X_dir_name, self.__y_dir_name = X_dir_name, y_dir_name
		self.__test_split_size = test_split_size
		self.__split_shuffle = split_shuffle

		self.__ma_layer = MovingAverage(self.__ma_window)

	@staticmethod
	def __generate_filename() -> str:
		return f"{datetime.now().timestamp()}.npy"

	@staticmethod
	def __save_array(arr: np.ndarray, path: str):
		np.save(path, arr)

	def __ma(self, sequence: np.ndarray):
		return self.__ma_layer(
			torch.from_numpy(sequence.astype(np.float32))
		).detach().numpy()

	def __setup_save_path(self, path: str):
		if not os.path.exists(path):
			os.mkdir(path)

		role_paths = [
			os.path.join(path, role_dir_name)
			for role_dir_name in [self.__train_dir_name, self.__test_dir_name]
		]
		storage_paths = [
			os.path.join(role_path, storage_dir_name)
			for role_path in role_paths
			for storage_dir_name in [self.__X_dir_name, self.__y_dir_name]
		]

		for storage_path in storage_paths:
			if not os.path.exists(storage_path):
				os.makedirs(storage_path)

	def __save(
			self,
			X: np.ndarray,
			y: np.ndarray,
			path: str,
			is_test: bool
	) -> str:
		filename = self.__generate_filename()
		role_dir_name = self.__train_dir_name
		if is_test:
			role_dir_name = self.__test_dir_name

		role_path = os.path.join(path, role_dir_name)

		for arr, dir_name in zip(
				[X, y],
				[self.__X_dir_name, self.__y_dir_name]
		):
			save_path = os.path.join(role_path, dir_name, filename)
			self.__save_array(arr, save_path)

		return filename

	def __split_and_save(self, X: np.ndarray, y: np.ndarray, path: str):
		data_len = X.shape[0]

		indices = train_test_split(np.arange(data_len), test_size=self.__test_split_size, shuffle=self.__split_shuffle)

		for role_indices, is_test in zip(indices, [False, True]):
			self.__save(
				X=X[role_indices],
				y=y[role_indices],
				path=path,
				is_test=is_test
			)

	def __process_set(self, X: np.ndarray, y: np.ndarray):
		X = np.concatenate([self.__ma(X[:, :self.__seq_len]), X[:, self.__seq_len:]], axis=1)
		self.__split_and_save(X, y, self.__export_path)

	def start(
			self,
			input_path: str,
	):

		self.__setup_save_path(self.__export_path)
		filenames = os.listdir(os.path.join(input_path, self.__X_dir_name))
		for i, filename in enumerate(filenames):
			X, y = [np.load(os.path.join(input_path, inner_path, filename)) for inner_path in [self.__X_dir_name, self.__y_dir_name]]
			self.__process_set(X, y)
			print(f"Completed {(i+1)*100/len(filenames) :.2f}%...", end="\r")
