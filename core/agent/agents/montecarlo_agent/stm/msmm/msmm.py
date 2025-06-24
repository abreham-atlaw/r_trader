from abc import ABC, abstractmethod

import numpy as np

from core.environment.trade_state import MarketState
from lib.utils.stm import CueMemoryMatcher


class MarketStateMemoryMatcher(CueMemoryMatcher, ABC):

	@abstractmethod
	def _compare_instrument_state(self, cue: np.ndarray, memory: np.ndarray) -> bool:
		pass

	def _compare_instruments(self, cue: MarketState, memory: MarketState) -> bool:
		return np.all([
			self._compare_instrument_state(cue.get_state_of(*instrument), memory.get_state_of(*instrument))
			for instrument in cue.get_tradable_pairs()
		])

	def is_match(self, cue: MarketState, memory: MarketState) -> bool:
		return self._compare_instruments(cue, memory)
