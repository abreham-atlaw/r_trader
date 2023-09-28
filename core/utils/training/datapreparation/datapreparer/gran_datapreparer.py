import typing

import numpy as np

from core.utils.training.datapreparation import CombinedDataPreparer


class GranDataPreparer(CombinedDataPreparer):

	def __init__(self, seq_len: int,  percentages: typing.List[float], *args, **kwargs):
		super().__init__(seq_len+1, *args, column_headers=[f"X{i}" for i in range(seq_len)] + [f"y{i}" for i in range(len(percentages) + 1)], **kwargs)
		self.__percentages = percentages

	def __one_hot_encoding(self, value: int) -> np.ndarray:
		encoded = np.zeros((len(self.__percentages) + 1))
		encoded[value] = 1
		return encoded

	def __find_gap_index(self, number) -> int:
		for i in range(len(self.__percentages)):
			if number < self.__percentages[i]:
				return i  # Return the index of the gap
		return len(self.__percentages)  # Return None if no gap is found

	def _finalize_sequence(self, sequences: np.ndarray) -> np.ndarray:
		out_encodings = np.zeros((sequences.shape[0], len(self.__percentages) + 1))

		for i, sequence in enumerate(sequences):
			value = self.__find_gap_index(sequence[-1]/sequence[-2])
			out_encodings[i] = self.__one_hot_encoding(value)

		data = np.concatenate((sequences[:, :-1], out_encodings), axis=1)
		return data
