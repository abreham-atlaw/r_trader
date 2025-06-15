import typing
from abc import ABC, abstractmethod

import unittest
import numpy as np

from core import Config
from core.di import AgentUtilsProvider
from core.agent.agents.montecarlo_agent.stm.msmm import MarketStateMemoryMatcher
from core.environment.trade_state import MarketState


class AbstractMarketStateMemoryMatcherTest(unittest.TestCase, ABC):

	@abstractmethod
	def init_matcher(self) -> MarketStateMemoryMatcher:
		pass

	def _get_original_sequence(self) -> np.ndarray:
		X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/6/train/X/1740913843.59131.npy")
		return X[0, :-124]

	@staticmethod
	def __apply_p(sequence, p: float) -> np.ndarray:
		new_sequence = sequence.copy()
		new_sequence[:-1] = sequence[1:]
		new_sequence[-1] = sequence[-1] * p
		return new_sequence

	def _get_memory(self, sequence: np.ndarray, bound: typing.Tuple[float, float]) -> np.ndarray:
		return self.__create_state(self.__apply_p(sequence, np.mean(bound)))

	def _get_valid_cue(self, sequence: np.ndarray, bound: typing.Tuple[float, float]) -> np.ndarray:
		return self.__create_state(self.__apply_p(sequence, np.random.uniform(bound[0], bound[1])))

	def _get_invalid_cue(self, sequence: np.ndarray, bound: typing.Tuple[float, float]) -> np.ndarray:
		return self.__create_state(self.__apply_p(sequence, bound[1] + self.noise))

	def _get_test_bounds(self) -> typing.List[typing.Tuple[float, float]]:
		idxs = np.random.randint(0, self.bounds.shape[0]-1, size=3)
		return [(self.bounds[idx], self.bounds[idx + 1]) for idx in idxs]

	def setUp(self):
		self.bounds = np.array(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND)
		self.bounds_diff = self.bounds[1:] - self.bounds[:-1]
		self.threshold = np.min(self.bounds_diff)/2
		self.noise = self.threshold * 0.01
		self.matcher = self.init_matcher()

		self.bounds = self._get_test_bounds()
		original_sequence = self._get_original_sequence()
		self.memories = [
			self._get_memory(original_sequence, bound)
			for bound in self.bounds
		]
		self.valid_cues = [
			self._get_valid_cue(original_sequence, bound)
			for bound in self.bounds
		]
		self.invalid_cues = [
			self._get_invalid_cue(original_sequence, bound)
			for bound in self.bounds
		]

	def __create_state(self, state: np.ndarray) -> MarketState:

		market_state = MarketState(
			currencies=["AUD", "USD"],
			memory_len=state.shape[0],
			tradable_pairs=[
				("AUD", "USD"),
			]
		)

		market_state.update_state_of("AUD", "USD", state)

		return market_state

	def test_valid(self):

		for i in range(len(self.bounds)):
			memory, cue = self.memories[i], self.valid_cues[i]
			is_match = self.matcher.is_match(cue, memory)
			self.assertTrue(is_match)

	def test_invalid(self):

		for i in range(len(self.bounds)):
			memory, cue = self.memories[i], self.invalid_cues[i]
			self.assertFalse(self.matcher.is_match(cue, memory))
