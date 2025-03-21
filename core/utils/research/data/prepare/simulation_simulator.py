import os
import typing
from datetime import datetime

import pandas as pd
import numpy as np

from lib.utils.logger import Logger
from lib.utils.math import moving_average


class SimulationSimulator:

	def __init__(
			self,
			dfs: typing.Union[pd.DataFrame, typing.List[pd.DataFrame]],
			bounds: typing.List[float],
			seq_len: int,
			extra_len: int,
			batch_size: int,
			output_path: str,
			granularity: int,
			ma_window: int = None,
			order_gran: bool = True
	):
		if isinstance(dfs, pd.DataFrame):
			dfs = [dfs]
		self.__dfs = self.__common_out_dfs([self.__process_df(df) for df in dfs])
		self.__bounds = bounds
		self.__seq_len = seq_len
		self.__extra_len = extra_len
		self.__batch_size = batch_size
		self.__output_path = output_path
		self.__granularity = granularity
		self.__ma_window = ma_window
		self.__order_gran = order_gran

	@property
	def __use_ma(self):
		return self.__ma_window not in [None, 0, 1]

	@staticmethod
	def __process_df(df: pd.DataFrame) -> pd.DataFrame:
		df["time"] = pd.to_datetime(df["time"])
		df = df.drop_duplicates(subset="time")
		df = df.sort_values(by="time")
		return df

	@staticmethod
	def __get_common_datetimes(dfs: typing.List[pd.DataFrame]) -> typing.List[datetime]:
		dates = np.concatenate([
			df["time"].to_numpy()
			for df in dfs
		])
		unique_dates, counts = np.unique(dates, return_counts=True)
		return unique_dates[counts == len(dfs)]

	def __common_out_dfs(self, dfs: typing.List[pd.DataFrame]) -> typing.List[pd.DataFrame]:
		if len(dfs) == 1:
			return dfs
		common_dates = self.__get_common_datetimes(dfs)
		Logger.info(f"Found {len(common_dates)} common dates.")
		return [df[df["time"].isin(common_dates)] for df in dfs]

	def __find_gap_index(self, number: float) -> int:
		boundaries = self.__bounds
		for i in range(len(boundaries)):
			if number < boundaries[i]:
				return i
		return len(boundaries)

	@staticmethod
	def __one_hot_encode(classes: np.ndarray, length: int) -> np.ndarray:
		encoding = np.zeros((classes.shape[0], length))
		for i in range(classes.shape[0]):
			encoding[i, classes[i]] = 1
		return encoding

	@staticmethod
	def __generate_filename() -> np.ndarray:
		return f"{datetime.now().timestamp()}.npy"

	def __filter_df(self, df: pd.DataFrame, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
		Logger.info(f"Filtering Dataframe...")
		if start_date is not None:
			df = df[df["time"] >= start_date]
		if end_date is not None:
			df = df[df["time"] <= end_date]
		return df

	def __stack(self, sequence: np.ndarray) -> np.ndarray:
		Logger.info(f"Stacking Sequence...\n\n")
		length = self.__seq_len + 1
		stack = np.zeros((sequence.shape[0] - length + 1, length))
		for i in range(stack.shape[0]):
			stack[i] = sequence[i: i + length]
			Logger.info(f"{(i + 1) * 100 / stack.shape[0] :.2f}", end="\r")
		return stack

	def __prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		Logger.info(f"Preparing X...")
		return np.concatenate(
			(
				sequences[:, :-1],
				np.zeros((sequences.shape[0], self.__extra_len))
			),
			axis=1
		)

	def __prepare_y(self, sequence: np.ndarray) -> np.ndarray:
		Logger.info(f"Preparing y...")
		percentages = sequence[:, -1] / sequence[:, -2]
		classes = np.array([self.__find_gap_index(percentages[i]) for i in range(percentages.shape[0])])
		encoding = self.__one_hot_encode(classes, len(self.__bounds) + 1)
		return np.concatenate(
			(
				encoding,
				np.zeros((encoding.shape[0], 1))
			),
			axis=1
		)

	def __batch_array(
			self,
			X: np.ndarray,
			y: np.ndarray
	) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
		Logger.info(f"Batching Array...")
		size = X.shape[0]

		X_batches = []
		y_batches = []

		for i in range(0, size, self.__batch_size):
			X_batches.append(X[i: i + self.__batch_size])
			y_batches.append(y[i: i + self.__batch_size])

		return X_batches, y_batches

	def __save(self, X: np.ndarray, y: np.ndarray):
		filename = self.__generate_filename()

		for arr, path in zip((X, y), ("X", "y")):
			save_path = os.path.join(self.__output_path, path, filename)
			if not os.path.exists(os.path.dirname(save_path)):
				os.makedirs(os.path.dirname(save_path))
			np.save(save_path, arr)

	def __save_batches(self, X_batches, y_batches):
		Logger.info(f"Saving Batches...")
		for i, (X, y) in enumerate(zip(X_batches, y_batches)):
			self.__save(X, y)
			Logger.info(f"{(i+1) * 100 / len(X_batches) :.2f}", end="\r")

	def __prepare_sequence(self, sequence: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
		stacked_sequence = self.__stack(sequence)

		X = self.__prepare_x(stacked_sequence)
		y = self.__prepare_y(stacked_sequence)

		return X, y

	def __concatenate_grans(self, arrays: typing.List[np.ndarray]) -> np.ndarray:

		def p(g, i, G):
			return g + G*i

		if not self.__order_gran:
			return np.concatenate(arrays, axis=0)
		new_arr = np.zeros((sum(arr.shape[0] for arr in arrays), arrays[0].shape[1]))
		for i in range(len(arrays)):
			new_arr[p(i, np.arange(arrays[i].shape[0]), len(arrays))] = arrays[i]
		return new_arr

	def __prepare_data(self, df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
		Logger.info(f"Preparing Data...")

		Xs, ys = [], []

		for i in range(self.__granularity):
			gran_sequence = df["c"].to_numpy()[i::self.__granularity]
			if self.__use_ma:
				gran_sequence = moving_average(gran_sequence, self.__ma_window)
			gran_X, gran_y = self.__prepare_sequence(gran_sequence)

			Xs.append(gran_X)
			ys.append(gran_y)

		X, y = [self.__concatenate_grans(arrays) for arrays in [Xs, ys]]
		return X, y

	def __merge_instruments_data(self, data: typing.List[typing.Tuple[np.ndarray, np.ndarray]]) -> typing.Tuple[np.ndarray, np.ndarray]:

		def merge_arrays(arrays: typing.List[np.ndarray]) -> np.ndarray:
			return np.concatenate([
				np.expand_dims(a, axis=1)
				for a in arrays
			], axis=1)

		if len(data) == 1:
			return data[0]

		X, y = [
			merge_arrays([
				instrument_data[axis]
				for instrument_data in data
			])
			for axis in [0, 1]
		]
		return X, y

	def start(self, start_date: datetime = None, end_date: datetime = None):
		dfs = [
			self.__filter_df(df, start_date, end_date)
			for df in self.__dfs
		]
		data = [
			self.__prepare_data(df)
			for df in dfs
		]
		X, y = self.__merge_instruments_data(data)
		X_batches, y_batches = self.__batch_array(X, y)
		self.__save_batches(X_batches, y_batches)
