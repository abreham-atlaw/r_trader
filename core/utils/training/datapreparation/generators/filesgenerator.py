from typing import *

import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence as KerasSequence
from sklearn.utils import shuffle as sk_shuffle

import random
import gc

from core.utils.training.datapreparation.cache import Cache


class FilesGenerator(KerasSequence):

	def __init__(
			self, 
			files: List[str],
			batch_size: int,
			max_row: int = 1000000,
			file_cache_size: int = 1,
			batch_cache_size: int = 1,
			random_seed: int = 0
	):
		self.__files = sorted(files)
		random.Random(random_seed).shuffle(self.__files)
		self.batch_size = batch_size
		self.__max_row = max_row
		self.__files_cache = Cache(file_cache_size)
		self.__batch_cache = Cache(batch_cache_size)
		self.__size = int(np.ceil(self.__get_total_num_rows() / self.batch_size))
		self.__random_state = self.__get_new_random_state()

	def __load_file(self, idx, shuffle=False) -> pd.DataFrame:
		cached: Optional[pd.DataFrame] = self.__files_cache.retrieve(idx)
		if cached is not None:
			return cached
		print("[+]Loading %s..." % (self.__files[idx]))
		df = pd.read_csv(self.__files[idx])
		self.__files_cache.store(idx, df)

		gc.collect()

		if shuffle:
			df = sk_shuffle(df, random_state=self.__random_state)
			df.reset_index(drop=True, inplace=True)

		return df

	def __separate_input_output(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
		return df.drop(columns=[self.__y_column_header]).to_numpy(), df[self.__y_column_header].to_numpy()

	def __get_indexes(self, idx: int) -> Tuple[int, int]:
		return idx // self.__max_row, idx % self.__max_row

	def __get_slice(self, start: int, end: int) -> pd.DataFrame:
		start_file_index, start_row_index = self.__get_indexes(start)
		end_file_index, end_row_index = self.__get_indexes(end)

		if start_file_index == end_file_index:
			return self.__load_file(start_file_index).iloc[start_row_index: end_row_index]

		batch_start = self.__load_file(start_file_index).iloc[start_row_index:]
		if end_file_index > len(self.__files) - 1:
			return batch_start
		batch_end = self.__load_file(end_file_index).iloc[:end_row_index]
		return pd.concat([batch_start, batch_end])

	def __getitem__(self, index):
		cached = self.__batch_cache.retrieve(index)
		if cached is not None:
			return cached

		batch = self.__get_slice(
			self.batch_size * index,
			self.batch_size * (index + 1)
		).to_numpy()
		self.__batch_cache.store(index, batch)
		return batch

	def __get_total_num_rows(self) -> int:
		last_file_size = len(self.__load_file(len(self.__files) - 1, shuffle=False))
		return (self.__max_row * len(self.__files[:-1])) + last_file_size

	def __get_new_random_state(self) -> int:
		return np.random.randint(len(self.__files) * 100)

	def on_epoch_end(self):
		self.__random_state = self.__get_new_random_state()

	def __len__(self):
		return self.__size
