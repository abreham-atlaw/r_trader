import typing

import numpy as np

import os
from datetime import datetime
import random

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

			split_shuffle: bool = True,
			batch_size: int = None

	):
		self.__input_size = input_size
		self.__seq_len = seq_len
		self.__ma_window = ma_window
		self.__export_path = export_path
		self.__batch_size = batch_size

		self.__train_dir_name, self.__test_dir_name = train_dir_name, test_dir_name
		self.__X_dir_name, self.__y_dir_name = X_dir_name, y_dir_name
		self.__test_split_size = test_split_size
		self.__split_shuffle = split_shuffle

		self.__ma_layer = MovingAverage(self.__ma_window) if self.__use_ma else None

	@property
	def __use_ma(self) -> bool:
		return self.__ma_window > 1

	@property
	def __is_batched(self) -> bool:
		return self.__batch_size is not None

	@staticmethod
	def __generate_filename() -> str:
		return f"{datetime.now().timestamp()}.npy"

	@staticmethod
	def __save_array(arr: np.ndarray, path: str):
		np.save(path, arr)

	def __batch_array(self, array: np.ndarray) -> typing.List[np.ndarray]:
		arrays = np.array_split(array, self.__batch_size)
		for i in range(len(arrays)):
			if arrays[i].shape[0] > self.__batch_size:
				arrays[i] = arrays[i][:self.__batch_size]
		return arrays

	def __ma(self, sequence: np.ndarray):
		if self.__use_ma:
			return sequence
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
		if X.shape[0] == 0:
			return
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

	def __batch_and_save(self, X: np.ndarray, y: np.ndarray, path: str, is_test: bool):
		if not self.__is_batched:
			self.__save(
				X=X,
				y=y,
				path=path,
				is_test=is_test
			)
			return
		X_batches = self.__batch_array(X)
		y_batches = self.__batch_array(y)

		for X_batch, y_batch in zip(X_batches, y_batches):
			self.__save(
				X=X_batch,
				y=y_batch,
				path=path,
				is_test=is_test
			)

	def __split_and_save(self, X: np.ndarray, y: np.ndarray, path: str):
		data_len = X.shape[0]

		if self.__test_split_size == 0:
			indices = list(np.arange(X.shape[0])), []
			if self.__split_shuffle:
				random.shuffle(indices[0])

		else:
			indices = train_test_split(np.arange(data_len), test_size=self.__test_split_size, shuffle=self.__split_shuffle)

		for role_indices, is_test in zip(indices, [False, True]):
			self.__batch_and_save(
				X[role_indices],
				y[role_indices],
				path,
				is_test
			)

	def __process_set(self, X: np.ndarray, y: np.ndarray):
		if self.__use_ma:
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
