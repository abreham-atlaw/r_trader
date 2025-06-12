import typing

import pandas as pd
import numpy as np

import os
from datetime import datetime
import gc

from sklearn.model_selection import train_test_split

from core.utils.research.data.prepare import SimulationSimulator


class DataPreparer:

	def __init__(
			self,
			boundaries: typing.List[float],
			block_size: int,

			granularity=1,
			ma_window_size=1,
			batch_size=1e5,

			train_dir_name: str = "train",
			test_dir_name: str = "test",
			X_dir_name: str = "X",
			y_dir_name: str = "y",

			test_split_size: float = 0.2,
			verbose: bool = True
	):
		self.__ma_window_size = ma_window_size
		self.__granularity = granularity
		self.__block_size = block_size
		self.__batch_size = batch_size

		self.__train_dir_name, self.__test_dir_name = train_dir_name, test_dir_name
		self.__X_dir_name, self.__y_dir_name = X_dir_name, y_dir_name
		self.__test_split_size = test_split_size

		self.__boundaries = boundaries

		self.__verbose = verbose

	@staticmethod
	def __generate_filename() -> str:
		return f"{datetime.now().timestamp()}.npy"

	@staticmethod
	def __moving_average(sequence, window_size):
		weights = np.repeat(1.0, window_size) / window_size
		smas = np.convolve(sequence, weights, 'valid')
		return smas

	@staticmethod
	def __stack_sequence(sequence: np.ndarray, seq_len: int) -> np.ndarray:
		X = np.zeros((sequence.shape[0] - seq_len + 1, seq_len))
		for i in range(X.shape[0]):
			X[i] = sequence[i: i + seq_len]
		return X

	@staticmethod
	def __one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
		y_sm = np.zeros((y.shape[0], num_classes), dtype=np.int16)
		y_sm[np.arange(y.shape[0]), y] = 1
		return y_sm

	@staticmethod
	def __find_gap_index(number: int, boundaries: typing.List[int]) -> int:
		for i in range(len(boundaries)):
			if number < boundaries[i]:
				return i
		return len(boundaries)

	def __prepare_y(self, percentages: np.ndarray) -> np.ndarray:
		classes = np.array([self.__find_gap_index(percentages[i], self.__boundaries) for i in range(percentages.shape[0])])
		encoding = self.__one_hot_encode(classes, len(self.__boundaries) + 1)
		return encoding

	def __prepare_sequence(self, sequence: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
		stack = self.__stack_sequence(sequence, self.__block_size + 1)
		X, y = stack[:, :-1], self.__prepare_y(stack[:, -1] / stack[:, -2])
		return X, y

	@staticmethod
	def __save_array(arr: np.ndarray, path: str):
		np.save(path, arr)

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

		indices = train_test_split(np.arange(data_len), test_size=self.__test_split_size)

		for role_indices, is_test in zip(indices, [False, True]):
			self.__save(
				X=X[role_indices],
				y=y[role_indices],
				path=path,
				is_test=is_test
			)

	def __setup_save_path(
			self, path: str):
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

	def __checkpoint(self, X: np.ndarray, y: np.ndarray, save_path: str) -> typing.Tuple[
		np.ndarray, np.ndarray]:
		size = X.shape[0]
		if self.__batch_size is None or size < self.__batch_size:
			return X, y

		to_save = [arr[:self.__batch_size] for arr in [X, y]]
		self.__split_and_save(*to_save, path=save_path)
		del to_save
		gc.collect()

		remaining = [arr[self.__batch_size:] for arr in [X, y]]
		return self.__checkpoint(*remaining, save_path=save_path)

	def __print(self, *args, **kwargs):
		if self.__verbose:
			print(*args, **kwargs)

	def start(
			self,
			df: pd.DataFrame,
			save_path: str,
			header_close: str = "c",
			export_remaining: bool = True
	):
		if not export_remaining and self.__batch_size is None:
			print("Warning: export_remaining set to false without batch_size. No exports will be made.")

		df = SimulationSimulator.prepare_df(df)

		self.__setup_save_path(save_path)

		X, y = None, None

		sequence = df[header_close].to_numpy()

		self.__print("[+]Preparing...")

		for i in range(self.__granularity):
			gran_sequence = sequence[i::self.__granularity]
			if self.__ma_window_size not in [0, 1, None]:
				gran_sequence = self.__moving_average(gran_sequence, self.__ma_window_size)

			X_gran, y_gran = self.__prepare_sequence(gran_sequence)

			if X is None:
				X, y = X_gran, y_gran
			else:
				X, y = [
					np.concatenate([old, new], axis=0)
					for (old, new) in zip([X, y], [X_gran, y_gran])
				]
			del X_gran, y_gran
			gc.collect()

			X, y = self.__checkpoint(X, y, save_path)
			gc.collect()
			self.__print(f"[+]Preparing: {(i + 1) * 100 / df.shape[0] :.2f}% ...", end="\r")

		if export_remaining:
			self.__split_and_save(
				X,
				y,
				save_path
			)

