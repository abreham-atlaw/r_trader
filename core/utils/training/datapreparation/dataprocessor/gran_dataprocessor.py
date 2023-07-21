from typing import *

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence as KerasSequence

import hashlib
import gc

from core.utils.training.datapreparation.generators import WrapperGenerator
from core.utils.training.datapreparation.cache import Cache
from .combined_dataprocessor import BatchDepthCache


class GranularDataProcessor:

    def __init__(
            self,
            generator: KerasSequence,
            model: Model,
            bounds: List[float],
            mini_batch_size: int,
            process_batch_size: int,
            cache_size: int = 5,
            depth_input: bool = True
    ):
        self.__generator = generator
        self.__model = model
        self.__bounds = np.array(bounds)
        self.__mini_batch_size = mini_batch_size
        self.__process_batch_size = process_batch_size
        self.__cache = BatchDepthCache(cache_size)
        self.__depth_input = depth_input
        self.__seq_len = self.__model.input_shape[1]
        if depth_input:
            self.__seq_len -= 1

    def set_model(self, model: Model):
        self.__model = model

    def __forecast(self, sequence, depth, initial_depth=0) -> np.ndarray:

        for i in range(initial_depth, depth):
            inputs = sequence
            if self.__depth_input:
                inputs = np.concatenate((inputs, i), axis=1)
            probs = self.__model.predict(inputs)
            next_bound = np.array([np.random.choice(self.__bounds, p=prob) for prob in probs])
            values = sequence[:, -1] + next_bound
            sequence = np.concatenate((sequence[:, 1:], np.expand_dims(values, axis=1)), axis=1)

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

    def __process_batch(self, batch: np.ndarray, depth) -> Tuple[np.ndarray, np.ndarray]:

        input_sequence = batch[:, :self.__seq_len]
        if depth > 0:
            input_sequence = self.__apply_depth(input_sequence, depth)

        core_input_size = self.__seq_len
        if self.__depth_input:
            core_input_size += 1

        x = np.zeros((batch.shape[0], core_input_size))
        y = np.zeros((batch.shape[0], len(self.__bounds)))

        x[:, :self.__seq_len] = input_sequence

        if self.__depth_input:
            x[:, -1] = depth

        delta = batch[:, self.__seq_len + depth] - input_sequence[:, -1]
        y = np.eye(len(self.__bounds))[(np.abs(self.__bounds - delta[:, None])).argmin(axis=1)]

        return x, y

    def get_data(self, idx, depth) -> WrapperGenerator:
        generator = WrapperGenerator(self.__mini_batch_size)

        batch = self.__generator[idx]
        rounds = int(np.ceil(len(batch) / self.__process_batch_size))
        for i in range(rounds):
            batch_data = self.__process_batch(
                batch[i * self.__process_batch_size: (i + 1) * self.__process_batch_size], depth)
            generator.add_data(batch_data)
            gc.collect()

        return generator

    def __len__(self):
        return len(self.__generator)
