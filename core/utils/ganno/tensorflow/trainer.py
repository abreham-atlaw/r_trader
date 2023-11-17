from typing import *
from abc import ABC, abstractmethod

from tensorflow.keras.models import Model
import numpy as np
from datetime import datetime

from lib.network.oanda import Trader
from lib.utils import math as libmath
from lib.utils.logger import Logger


class Trainer(ABC):

	@abstractmethod
	def fit(self, core_model: Model, delta_model: Model) -> Tuple[Tuple[float, float], float]:
		pass

	@abstractmethod
	def evaluate(self, core_model: Model, delta_model: Model) -> float:
		pass


class LiveTrainer(Trainer):

	def __init__(
			self,
			trader: Trader,
			data_point_per_instrument: int,
			granularity: str = "M1",
			average_window: int = 10,
			batch_size: int = 16,
			epochs: int = 5,
			instruments: List[Tuple[str, str]] = None
	):
		self.__trader = trader
		self.__size = data_point_per_instrument
		self.__granularity = granularity
		self.__average_window = average_window
		self.__batch_size = batch_size
		self.__epochs = epochs
		self.__instruments = instruments
		if instruments is None:
			self.__instruments = self.__trader.get_instruments()

	def __get_sequence(self, instrument, size, gran) -> np.ndarray:
		candlesticks = self.__trader.get_candlestick(instrument, count=size, to=datetime.now(), granularity=gran)
		return np.array([float(point.mid["c"]) for point in candlesticks])

	def __generate_core_data_instrument(self, instrument: Tuple[str, str], seq_len: int):
		total_length = self.__size + seq_len + self.__average_window
		X = np.zeros((self.__size, seq_len+1))
		y = np.zeros((self.__size,))

		sequence = libmath.moving_average(
			self.__get_sequence(instrument, total_length, self.__granularity),
			self.__average_window
		)
		for i in range(self.__size):
			X[i:, :-1] = sequence[i: i+seq_len]
			if sequence[i+seq_len] > sequence[i+seq_len-1]:
				y[i] = 1
			else:
				y[i] = 0
		return X, y

	def __generate_core_data(self, seq_len: int):
		X = np.zeros((0, seq_len+1))
		y = np.zeros(0)
		for instrument in self.__instruments:
			ins_x, ins_y = self.__generate_core_data_instrument(instrument, seq_len)
			X = np.concatenate((X, ins_x))
			y = np.concatenate((y, ins_y))
		return X, y

	def __generate_delta_data_instrument(self, instrument: Tuple[str, str], seq_len: int):
		total_length = self.__size + seq_len + self.__average_window
		X = np.zeros((self.__size, seq_len+2))
		y = np.zeros((self.__size,))

		sequence = libmath.moving_average(
			self.__get_sequence(instrument, total_length, self.__granularity),
			self.__average_window
		)
		for i in range(self.__size):
			direction = 0
			if sequence[i + seq_len] > sequence[i + seq_len - 1]:
				direction = 1
			X[i, :-2] = sequence[i: i + seq_len]
			X[i, -2] = direction
			y[i] = np.abs(sequence[i+seq_len] - sequence[i+seq_len-1])
		return X, y

	def __generate_delta_data(self, seq_len: int):
		X = np.zeros((0, seq_len+2))
		y = np.zeros(0)
		for instrument in self.__instruments:
			ins_x, ins_y = self.__generate_delta_data_instrument(instrument, seq_len)
			X = np.concatenate((X, ins_x))
			y = np.concatenate((y, ins_y))
		return X, y

	@Logger.logged_method
	def __fit_core_model(self, core_model: Model) -> Tuple[float, float]:
		X, y = self.__generate_core_data(core_model.input_shape[1] - 1)
		history = core_model.fit(X, y, batch_size=self.__batch_size, epochs=self.__epochs)
		return history.history["loss"], history.history["accuracy"]

	@Logger.logged_method
	def __fit_delta_model(self, delta_model: Model) -> float:
		X, y = self.__generate_delta_data(delta_model.input_shape[1] - 2)
		history = delta_model.fit(X, y, batch_size=self.__batch_size, epochs=self.__epochs)
		return history.history["loss"]

	def __evaluate_core_model(self, model: Model) -> float:
		X, y = self.__generate_core_data(model.input_shape[1]-1)
		results = model.evaluate(X, y, batch_size=self.__batch_size)
		return results[0]

	def __evaluate_delta_model(self, model: Model) -> float:
		X, y = self.__generate_delta_data(model.input_shape[1] - 2)
		results = model.evaluate(X, y, batch_size=self.__batch_size)
		return results

	def fit(self, core_model: Model, delta_model: Model) -> Tuple[Tuple[float, float], float]:
		return self.__fit_core_model(core_model), self.__fit_delta_model(delta_model)

	def evaluate(self, core_model: Model, delta_model: Model) -> float:
		return float(np.mean([self.__evaluate_core_model(core_model), self.__evaluate_delta_model(delta_model)]))
