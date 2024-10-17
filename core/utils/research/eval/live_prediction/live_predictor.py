import time
import typing

import matplotlib.pyplot as plt
import numpy as np

from core import Config
from core.Config import OANDA_TOKEN, OANDA_TEST_ACCOUNT_ID
from core.utils.research.model.model.utils import WrappedModel
from lib.network.oanda import Trader
from lib.rl.agent.dta import TorchModel
from lib.utils.torch_utils.model_handler import ModelHandler


class LivePredictor:

	def __init__(
			self,
			model_path: str,
			prediction_window=5,
			top_k=100,
			top_k_final=20,
			instrument: typing.Tuple[str, str] = ("AUD", "USD"),
			gran="M5",
			seq_len=Config.MARKET_STATE_MEMORY,
			window_size=Config.AGENT_MA_WINDOW_SIZE,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			extra_len=124,
			sleep_time=1,
			display_window=32,
			delay_mode=True,
			y_lim=1e-3
	):
		self.__model = TorchModel(WrappedModel(
					ModelHandler.load(model_path),
					seq_len=seq_len,
					window_size=window_size
				))
		self.__instrument = instrument
		self.__gran = gran
		self.__trader = Trader(OANDA_TOKEN, OANDA_TEST_ACCOUNT_ID)
		self.__seq_len = seq_len + window_size - 1
		self.__extra_len = extra_len
		self.__bounds = bounds
		self.__top_k = top_k
		self.__top_k_final = top_k_final
		self.__prediction_widow = prediction_window
		self.__sleep_time = sleep_time
		self.__display_window = display_window
		self.__delay_mode = delay_mode
		self.__sequence_cache = None
		self.__y_lim = y_lim

	def __predict(self, sequence: np.ndarray) -> np.ndarray:
		input_data = np.concatenate((sequence[:, -self.__seq_len:], np.zeros((sequence.shape[0], self.__extra_len))), axis=1)
		prediction = self.__model(input_data)
		probability_distribution = prediction[:, :-2].flatten()
		return probability_distribution

	def __fetch_sequence(self) -> np.ndarray:
		count = self.__seq_len
		if self.__delay_mode:
			count += self.__prediction_widow
		self.__sequence_cache = np.array([
			float(cs.mid["c"])
			for cs in self.__trader.get_candlestick(self.__instrument, count=count, granularity=self.__gran)
		])
		return self.__sequence_cache

	def __generate_simulation(self, sequence: np.ndarray) -> np.ndarray:
		simulation = np.repeat(sequence, len(self.__bounds), axis=0)
		simulation = np.concatenate(
			(
				simulation,
				simulation[:, -1:] * np.concatenate([np.array(self.__bounds) for _ in range(sequence.shape[0])], axis=0).reshape((-1, 1))
			),
			axis=1
		)
		return simulation

	def __generate_probability_distribution(self, sequence: np.ndarray, probability_distribution: np.ndarray) -> np.ndarray:
		probability_distribution = np.repeat(probability_distribution, len(self.__bounds), axis=0)
		new_pd = self.__predict(sequence)
		return probability_distribution * np.expand_dims(new_pd, axis=1)

	def __select_top(self, sequence: np.ndarray, probability_distribution: np.ndarray, top_k=None) -> typing.Tuple[np.ndarray, np.ndarray]:
		if top_k is None:
			top_k = self.__top_k
		selected_idx = np.argsort(probability_distribution.flatten())[::-1][:top_k]

		return sequence[selected_idx], probability_distribution[selected_idx]

	def __simulate_timestep(self, sequence: np.ndarray, probability_distribution) -> typing.Tuple[np.ndarray, np.ndarray]:
		simulation = self.__generate_simulation(sequence)
		probability_distribution = self.__generate_probability_distribution(sequence, probability_distribution)
		simulation, probability_distribution = self.__select_top(simulation, probability_distribution)
		return simulation, probability_distribution

	def __simulate(self, sequence: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
		sequence = np.expand_dims(sequence, axis=0)
		probability_distribution = np.ones((1, 1))
		for i in range(self.__prediction_widow):
			sequence, probability_distribution = self.__simulate_timestep(sequence, probability_distribution)
		sequence, probability_distribution = self.__select_top(sequence, probability_distribution, top_k=self.__top_k_final)
		return sequence, probability_distribution

	def __get_and_simulate(self):
		sequence = self.__fetch_sequence()
		if self.__delay_mode:
			sequence = sequence[:-self.__prediction_widow]
		simulation, probability_distribution = self.__simulate(sequence)
		return simulation, probability_distribution

	def plot_timestep(self):

		sequence, pd = self.__get_and_simulate()
		pd = pd * 2 / np.max(pd)

		plt.clf()
		for series, p in zip(sequence, pd.flatten()):
			plt.plot(series[-self.__display_window:], linewidth=p)

		x = self.__display_window - self.__prediction_widow - 1
		y = sequence[0, -self.__display_window:][x]

		if self.__delay_mode:
			true_sequence = self.__sequence_cache
			plt.plot(true_sequence[-self.__display_window:], linewidth=2, color="red")
			plt.scatter([x], [y], s=100, c="red")

		plt.ylim(y - self.__y_lim, y+self.__y_lim)

		plt.draw()
		plt.pause(self.__sleep_time)

	def start(self):
		plt.ion()
		plt.figure()
		try:
			while True:
				self.plot_timestep()
		except KeyboardInterrupt:
			plt.ioff()
			plt.show()
