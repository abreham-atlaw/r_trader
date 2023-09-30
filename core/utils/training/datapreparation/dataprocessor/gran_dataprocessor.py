import typing

from tensorflow import keras
import numpy as np

from .dataprocessor import DataProcessor


class GranDataProcessor(DataProcessor):

	def __init__(
			self,
			grans: typing.List[float],
			model: keras.Model,
			*args, **kwargs
	):
		super().__init__(model, None, *args, **kwargs)
		self.__percentages = grans
		self.__classes = self.__get_classes(grans)

	@staticmethod
	def __get_classes(percentages):
		averages = [percentages[0]]
		for i in range(len(percentages) - 1):
			average = (percentages[i] + percentages[i + 1]) / 2.0
			averages.append(average)
		averages.append(percentages[-1])
		return np.array(averages)

	def __one_hot_encoding(self, value: int) -> np.ndarray:
		encoded = np.zeros((len(self.__percentages) + 1))
		encoded[value] = 1
		return encoded

	def __find_gap_index(self, number) -> int:
		for i in range(len(self.__percentages)):
			if number < self.__percentages[i]:
				return i
		return len(self.__percentages)

	def __encode_gap_indexes(self, array: np.ndarray) -> np.ndarray:
		indexes = np.zeros((array.shape[0], len(self.__classes)))
		for i in range(indexes.shape[0]):
			indexes[i] = self.__one_hot_encoding(self.__find_gap_index(array[i]))
		return indexes

	def _forecast(self, sequence, depth, initial_depth=0) -> np.ndarray:

		for i in range(initial_depth, depth):
			depth = np.ones((sequence.shape[0], 1)) * i

			core_input = sequence
			if self._depth_input:
				core_input = np.concatenate((core_input, depth), axis=1)

			grans = self._core_model.predict(core_input, verbose=0)
			classes = np.argmax(grans, axis=1)
			percentages = self.__classes[classes]
			values = sequence[:, -1] * percentages
			sequence = np.concatenate((sequence[:, 1:], np.expand_dims(values, axis=1)), axis=1)

		return sequence

	def _process_batch(self, batch: np.ndarray, depth) -> typing.Tuple[typing.Tuple[np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]]:

		input_sequence = batch[:, :self._seq_len]
		if depth > 0:
			input_sequence = self._apply_depth(input_sequence, depth)

		core_input_size = self._seq_len
		if self._depth_input:
			core_input_size += 1

		core_x, core_y = np.zeros((batch.shape[0], core_input_size)), np.zeros((batch.shape[0],))

		core_x[:, :self._seq_len] = input_sequence

		if self._depth_input:
			core_x[:, -1] = depth

		core_y = self.__encode_gap_indexes(
			batch[:, self._seq_len + depth] / input_sequence[:, -1]
		)

		return (core_x, core_y), (np.zeros((1, 1)), np.zeros((1,)))
