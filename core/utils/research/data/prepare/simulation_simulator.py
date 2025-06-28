import os
import typing
from datetime import datetime

import pandas as pd
import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm, MovingAverage
from lib.utils.logger import Logger
from lib.utils.math import moving_average


class SimulationSimulator:

	def __init__(
			self,
			df: pd.DataFrame,
			bounds: typing.List[float],
			seq_len: int,
			extra_len: int,
			batch_size: int,
			output_path: str,
			granularity: int,
			ma_window: int = None,
			order_gran: bool = True,
			smoothing_algorithm: typing.Optional[SmoothingAlgorithm] = None
	):
		self.__df = self.prepare_df(df)
		self.__bounds = bounds
		self.__seq_len = seq_len
		self.__extra_len = extra_len
		self.__batch_size = batch_size
		self.__output_path = output_path
		self.__granularity = granularity
		self.__order_gran = order_gran

		if smoothing_algorithm is None and ma_window not in [None, 0, 1]:
			smoothing_algorithm = MovingAverage(ma_window)
		self.__smoothing_algorithm = smoothing_algorithm
		Logger.info(f"Using Smoothing Algorithm: {self.__smoothing_algorithm}")

	@property
	def __use_smoothing(self):
		return self.__smoothing_algorithm is not None

	@staticmethod
	def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
		df["time"] = pd.to_datetime(df["time"])
		df = df.drop_duplicates(subset="time")
		df = df.sort_values(by="time")
		return df

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

	def __filter_df(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
		Logger.info(f"Filtering Dataframe...")
		df = self.__df
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
		Logger.info(f"Saving Batches to {self.__output_path}...")
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
			if self.__use_smoothing:
				gran_sequence = self.__smoothing_algorithm(gran_sequence)
			gran_X, gran_y = self.__prepare_sequence(gran_sequence)

			Xs.append(gran_X)
			ys.append(gran_y)

		X, y = [self.__concatenate_grans(arrays) for arrays in [Xs, ys]]
		return X, y

	def start(self, start_date: datetime = None, end_date: datetime = None):
		df = self.__filter_df(start_date, end_date)
		X, y = self.__prepare_data(df)
		X_batches, y_batches = self.__batch_array(X, y)
		self.__save_batches(X_batches, y_batches)
