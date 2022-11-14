from typing import *
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from datetime import datetime
import os
import gc

from .filters import Filter


class CombinedDataPreparer:

	def __init__(
			self,
			seq_len: int,
			files: List[str],
			output_path: str,
			column_headers: List[str] = None,
			max_rows: int = 1000000,
			min_variations: int = 40,
			granularity: int = 1,
			batch_size: int = 10000,
			filters: Optional[List[Filter]] = None,
			export_remaining: bool = False,
			instrument_delimiter: str = "/",
	):
		self.__seq_len = seq_len
		self.__files = files
		self.__output_path = output_path
		self.__max_rows = max_rows
		self.__column_headers = column_headers
		if column_headers is None:
			self.__column_headers = [str(i) for i in range(seq_len)]
		self.__min_var = min_variations
		self.__granularity = granularity
		self.__batch_size = batch_size
		self.__filters = filters
		self.__export_remaining = export_remaining
		self.__instrument_delimiter = instrument_delimiter

	@staticmethod
	def __filter_instrument(df: pd.DataFrame, instrument: Tuple[str, str]) -> pd.DataFrame:
		df = df[df["base_currency"] == instrument[0]]
		df = df[df["quote_currency"] == instrument[1]]
		return df

	@staticmethod
	def _generate_filename() -> str:
		return f"{datetime.now().timestamp()}.csv"

	def __get_currency_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
		return [
			tuple(pair.split(self.__instrument_delimiter)[:2])
			for pair in set(df["base_currency"] + self.__instrument_delimiter + df["quote_currency"])
		]

	def __calc_filters_input_shape(self, output_shape: Tuple) -> Tuple:
		input_shape = output_shape
		for filter_ in self.__filters[::-1]:
			input_shape = filter_.calc_input_shape(input_shape)
		return input_shape

	def __apply_filters(self, X: np.ndarray) -> np.ndarray:
		for filter_ in self.__filters:
			X = filter_.filter(X)
		return X

	def __prepare_sequence(self, sequence: np.ndarray, seq_len: int) -> np.ndarray:
		if sequence.shape[0] < seq_len:
			raise DataSetTooSmall()
		X = np.zeros((sequence.shape[0] - seq_len + 1, seq_len))
		for i in range(X.shape[0]):
			X[i] = sequence[i: i + seq_len]
		return X

	def __prepare_df(self, df: pd.DataFrame) -> np.ndarray:
		instruments = self.__get_currency_pairs(df)

		X = np.zeros((0, self.__seq_len))
		pre_filter_size = self.__calc_filters_input_shape((0, self.__seq_len))[1]

		for base_currency, quote_currency in instruments:
			instrument_sequence = self.__filter_instrument(df, (base_currency, quote_currency))["c"].to_numpy()
			for i in range(self.__granularity):
				Xi = self.__prepare_sequence(instrument_sequence[i::self.__granularity], pre_filter_size)
				X_filtered = self.__apply_filters(Xi)
				X = np.concatenate((X, X_filtered))
				del Xi, X_filtered

		return X

	def __load_file(self, file_name: str, batch_idx: int) -> pd.DataFrame:
		return pd.read_csv(file_name, index_col=0).iloc[self.__batch_size * batch_idx: self.__batch_size * (batch_idx + 1)]

	def __process_file(self, file: str, batch_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

		raw_data = self.__load_file(file, batch_idx)
		if len(raw_data) < self.__calc_filters_input_shape((0, self.__seq_len))[1]:
			raise DataSetTooSmall()
		sequences = self.__prepare_df(raw_data)
		if len(sequences) == 0:
			df = pd.DataFrame(columns=self.__column_headers)
		else:
			df = pd.DataFrame(sequences, columns=self.__column_headers).dropna()
		return df

	def __create_df(self) -> pd.DataFrame:
		return pd.DataFrame(columns=self.__column_headers)

	def __save_df(self, df: pd.DataFrame, folder):
		df.to_csv(os.path.join(folder, self._generate_filename()), index=False)

	def __append_df(self, df: pd.DataFrame, new_data: pd.DataFrame, path: str) -> pd.DataFrame:
		if len(df) + len(new_data) < self.__max_rows:
			return df.append(new_data)
		bound = self.__max_rows - len(df)
		df = df.append(new_data.iloc[:bound])
		self.__save_df(df, path)

		return self.__append_df(self.__create_df(), new_data.iloc[bound:], path)

	def start(self):
		df = self.__create_df()

		batch_idx = 0
		while True:
			temp_df = self.__create_df()
			for i,  file in enumerate(self.__files):
				try:
					new_data = self.__process_file(file, batch_idx)
				except DataSetTooSmall:
					continue
				temp_df = temp_df.append(new_data)
				gc.collect()
				print(
					f"Processing Batch: {batch_idx + 1}\t\tDone:{100 * (i + 1) / len(self.__files) :.2f}%\t\tFinished: {file}",
					end="\r")

			if len(temp_df) == 0:
				break

			df = self.__append_df(
				df, temp_df, self.__output_path
			)

			batch_idx += 1

		if self.__export_remaining:
			self.__save_df(
				df,
				self.__output_path
			)


class DataSetTooSmall(Exception):
	pass
