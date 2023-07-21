from typing import *

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence as KerasSequence

import hashlib
import gc

from core.utils.training.datapreparation.generators import WrapperGenerator
from .cache import Cache


class BatchDepthCache(Cache):

	def __init__(self, size: int):
		super().__init__(size)

	@staticmethod
	def __generate_cache_key(batch, depth) -> str:
		return f"{hashlib.md5(batch.tobytes()).hexdigest()}-{depth}"

	def store(self, batch: np.ndarray, depth: int, data: np.ndarray):
		super().store(self.__generate_cache_key(batch, depth), data)

	def retrieve(self, batch, depth) -> Optional[np.ndarray]:
		return super().retrieve(self.__generate_cache_key(batch, depth))


class DataProcessor:

	def __init__(
			self,
			generator: KerasSequence,
			core_model: Model,
			delta_model: Model,
			mini_batch_size: int,
			process_batch_size: int,
			cache_size: int = 5,
			direction_rounding: bool = False,
			depth_input: bool = True,
			true_directions: bool = True
	):
		self.__generator = generator
		self.__core_model, self.__delta_model = core_model, delta_model
		self.__mini_batch_size = mini_batch_size
		self.__process_batch_size = process_batch_size
		self.__cache = BatchDepthCache(cache_size)
		self.__direction_rounding = direction_rounding
		self.__depth_input = depth_input
		self.__true_directions = true_directions
		self.__seq_len = self.__core_model.input_shape[1]
		if depth_input:
			self.__seq_len -= 1

	def set_models(self, core_model: Model, delta_model: Model):
		self.__core_model, self.__delta_model = core_model, delta_model

	def __forecast(self, sequence, depth, initial_depth=0) -> np.ndarray:

		for i in range(initial_depth, depth):
			depth = np.ones((sequence.shape[0], 1)) * i

			core_input = sequence
			if self.__depth_input:
				core_input = np.concatenate((core_input, i), axis=1)

			directions = self.__core_model.predict(core_input)

			round_directions = np.round(directions)

			delta_direction = directions
			if self.__direction_rounding:
				delta_direction = round_directions

			delta_input = np.concatenate((sequence, delta_direction), axis=1)
			if self.__depth_input:
				delta_input = np.concatenate((delta_input, i), axis=1)

			deltas = self.__delta_model.predict(delta_input)
			values = sequence[:, -1:] + (2 * round_directions - 1) * deltas
			sequence = np.concatenate((sequence[:, 1:], values), axis=1)

		return sequence

	def __apply_depth(self, sequence: np.ndarray, depth) -> np.ndarray:
		start_depth = 0
		input_sequence = sequence
		cached = self.__cache.retrieve(sequence, depth - 1)
		if cached is not None:
			input_sequence = cached
			start_depth = depth - 1
		forecast = self.__forecast(input_sequence, depth, initial_depth=start_depth)
		self.__cache.store(sequence, depth, forecast)
		return forecast

	def __process_batch(self, batch: np.ndarray, depth) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

		input_sequence = batch[:, :self.__seq_len]
		if depth > 0:
			input_sequence = self.__apply_depth(input_sequence, depth)

		core_input_size = self.__seq_len
		if self.__depth_input:
			core_input_size += 1

		core_x, core_y = np.zeros((batch.shape[0], core_input_size)), np.zeros((batch.shape[0],))
		delta_x, delta_y = np.zeros((batch.shape[0], core_input_size + 1)), np.zeros((batch.shape[0],))

		core_x[:, :self.__seq_len] = delta_x[:, :self.__seq_len] = input_sequence

		if self.__depth_input:
			core_x[:, -1] = delta_x[:, -1] = depth

		core_y[:] = delta_y[:] = batch[:, self.__seq_len + depth] - input_sequence[:, -1]
		core_y[core_y <= 0], core_y[core_y > 0] = 0, 1
		delta_y[:] = np.abs(delta_y)

		delta_x[:, self.__seq_len] = core_y
		if not self.__true_directions:
			delta_x[:, self.__seq_len] = self.__core_model.predict(core_x).flatten()
			if self.__direction_rounding:
				delta_x[:, self.__seq_len] = np.round(delta_x[:, -2])

		return (core_x, core_y), (delta_x, delta_y)

	def get_data(self, idx, depth) -> Tuple[WrapperGenerator, WrapperGenerator]:
		core_generator, delta_generator = WrapperGenerator(self.__mini_batch_size), WrapperGenerator(
			self.__mini_batch_size)

		batch = self.__generator[idx]
		rounds = int(np.ceil(len(batch) / self.__process_batch_size))
		for i in range(rounds):
			core_batch, delta_batch = self.__process_batch(
				batch[i * self.__process_batch_size: (i + 1) * self.__process_batch_size], depth)
			core_generator.add_data(core_batch)
			delta_generator.add_data(delta_batch)
			# print(f"\r[+]Preparing Data: {(i + 1) * 100 / rounds :.2f}%", end="")
			gc.collect()
		# print()

		return core_generator, delta_generator

	def __len__(self):
		return len(self.__generator)
