import hashlib
import typing

import numpy as np


class DuplicateCleaner:

	def __init__(self, arrays: typing.Iterable[np.ndarray] = None, hash_fn=None):
		if hash_fn is None:
			hash_fn = hashlib.md5
		self.__hash_fn = hash_fn
		if arrays is None:
			self.__hashes = np.array([])
		else:
			self.__hashes = self.__generate_hashes(arrays)

	@staticmethod
	def __filter_unique_rows(array: np.ndarray):
		array_tuple = [tuple(row) for row in array]

		clean_dict = {}
		for i, row in enumerate(array_tuple):
			if row not in clean_dict:
				clean_dict[row] = i

		clean_indices = np.array(list(clean_dict.values()))
		clean_array = array[clean_indices]

		return clean_array, clean_indices

	def __generate_hashes(self, arrays: typing.Iterable[np.ndarray]) -> np.ndarray:
		return np.concatenate([
			self.__hash(array)
			for array in arrays
		])

	def __hash(self, array) -> np.ndarray:
		return np.array([
			self.__hash_fn(array[i].tobytes()).hexdigest()
			for i in range(array.shape[0])
		])

	def __get_unique_rows(self, array: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
		checkpoint_size = self.__hashes.shape[0]
		hashes = self.__hash(array)
		complete_hashes = np.concatenate([self.__hashes, hashes])

		self.__hashes, indices = self.__filter_unique_rows(complete_hashes)
		indices = indices[checkpoint_size:] - checkpoint_size
		return array[indices], indices

	def clean(self, array: np.ndarray, companion_array: np.ndarray = None) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], np.ndarray]:
		clean_array, indices = self.__get_unique_rows(array)
		if companion_array is None:
			return clean_array
		return clean_array, companion_array[indices]
