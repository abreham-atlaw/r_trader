import os.path
import typing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from lib.utils.cache import Cache
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger


class SessionAnalyzer:

	def __init__(
			self,
			session_path: str,
			smoothing_algorithms: typing.List[SmoothingAlgorithm],
			instruments: typing.List[typing.Tuple[str, str]],
			fig_size: typing.Tuple[int, int] = (20, 10),
			plt_y_grid_count: int = 100,
	):
		self.__sessions_path = session_path
		self.__fig_size = fig_size
		self.set_smoothing_algorithms(smoothing_algorithms)
		self.__plt_y_grid_count = plt_y_grid_count
		self.__cache = Cache()
		self.__instruments = instruments

	@property
	def __candlesticks_path(self) -> str:
		return os.path.join(self.__sessions_path, "candlesticks")

	@property
	def __graphs_path(self) -> str:
		return os.path.join(self.__sessions_path, "graph_dumps")

	def __get_df_files(self, instrument: typing.Tuple[str, str]) -> typing.List[str]:

		idx = self.__instruments.index(instrument)
		all_files = [
			os.path.join(self.__candlesticks_path, file)
			for file in filter(
				lambda fn: fn.endswith(".csv"),
				sorted(os.listdir(self.__candlesticks_path))
			)
		]
		return all_files[idx::len(self.__instruments)]

	@CacheDecorators.cached_method()
	def __load_dfs(self, instrument: typing.Tuple[str, str]) -> pd.DataFrame:
		return list(filter(
			lambda df: df.shape[0] > 0,
			[
				pd.read_csv(os.path.join(self.__candlesticks_path, f))
				for f in self.__get_df_files(instrument)
			]
		))

	@CacheDecorators.cached_method()
	def __get_sequences(self, instrument: typing.Tuple[str, str]) -> typing.List[np.ndarray]:
		dfs = self.__load_dfs(instrument=instrument)
		return [
			df["c"].to_numpy()
			for df in dfs
		]

	@CacheDecorators.cached_method()
	def __get_smoothed_sequences(self, instrument: typing.List[str]) -> typing.List[typing.List[np.ndarray]]:
		x = self.__get_sequences(instrument)
		return [
			[
				sa(seq)
				for seq in x
			]
			for sa in self.__smoothing_algorithms
		]

	def set_smoothing_algorithms(self, smoothing_algorithms: typing.List[SmoothingAlgorithm]):
		self.__smoothing_algorithms = smoothing_algorithms
		Logger.info("Using Smoothing Algorithms: {}".format(', '.join([str(sa) for sa in smoothing_algorithms])))

	def plot_sequence(self, instrument: typing.Tuple[str, str], checkpoints: typing.List[int] = None):
		if checkpoints is None:
			checkpoints = []

		x = [
			seq[-1]
			for seq in self.__get_sequences(instrument=instrument)
		]
		x_sa = [
			[
				smoothed_sequence[-1]
				for smoothed_sequence in sa_sequences
			]
			for sa_sequences in self.__get_smoothed_sequences(instrument=instrument)
		]

		plt.figure(figsize=self.__fig_size)
		plt.title(" / ".join(instrument))
		plt.grid()

		plt.plot(x, label="Clean")
		for i in range(len(self.__smoothing_algorithms)):
			plt.plot(x_sa[i], label=str(self.__smoothing_algorithms[i]))

		for y in np.linspace(np.min(x), np.max(x), self.__plt_y_grid_count):
			plt.axhline(y=y, color="black")

		for checkpoint in checkpoints:
			plt.axvline(x=checkpoint, color="blue")
			plt.axvline(x=checkpoint+1, color="green")
			plt.axvline(x=checkpoint+2, color="red")
			plt.text(checkpoint, max(x), str(checkpoint), verticalalignment="center")

		plt.legend()
		plt.show()

	def plot_timestep_sequence(self, instrument: typing.Tuple[str, str], i: int):
		x = self.__get_sequences(instrument=instrument)[i]
		x_sa = [
			sequence[i]
			for sequence in self.__get_smoothed_sequences(instrument=instrument)
		]

		plt.figure(figsize=self.__fig_size)
		plt.title(f"{instrument[0]} / {instrument[1]}  -  i={i}")
		plt.grid()

		plt.plot(x, label="Clean")
		for i in range(len(self.__smoothing_algorithms)):
			plt.plot(x_sa[i], label=str(self.__smoothing_algorithms[i]))

		plt.legend()
		plt.show()
