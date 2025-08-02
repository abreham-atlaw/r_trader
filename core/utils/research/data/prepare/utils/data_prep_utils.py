import typing

import numpy as np
import pandas as pd

from lib.utils.logger import Logger


class DataPrepUtils:

	@staticmethod
	def find_bound_index(bounds: typing.List[float], value: float) -> int:
		return np.sum(value >= np.array(bounds))

	@staticmethod
	def apply_bound_epsilon(bounds: typing.List[float], eps: float = None) -> typing.List[float]:
		if eps is None:
			eps = (bounds[1] - bounds[0] + bounds[-1] - bounds[-2]) / 2
		Logger.info(f"Using epsilon: {eps}")
		bounds = np.concatenate([
			np.array([bounds[0] - eps]),
			bounds,
			np.array([bounds[-1] + eps])
		])
		bounds = (bounds[1:] + bounds[:-1]) / 2
		return bounds

	@staticmethod
	def clean_df(df: pd.DataFrame) -> pd.DataFrame:
		Logger.info(f"Cleaning DataFrame")
		df["time"] = pd.to_datetime(df["time"])
		df = df.drop_duplicates(subset="time")
		df = df.sort_values(by="time")
		return df

	@staticmethod
	def stack(sequence: np.ndarray, length) -> np.ndarray:
		stack = np.zeros((sequence.shape[0] - length + 1, length))
		for i in range(stack.shape[0]):
			stack[i] = sequence[i: i + length]
		return stack
