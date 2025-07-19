import typing

import numpy as np
import pandas as pd


class DataPrepUtils:

	@staticmethod
	def find_bound_index(bounds: typing.List[float], value: float) -> int:
		return np.sum(value >= np.array(bounds))

	@staticmethod
	def clean_df(df: pd.DataFrame) -> pd.DataFrame:
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
