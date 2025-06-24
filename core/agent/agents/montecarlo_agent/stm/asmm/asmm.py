from abc import ABC, abstractmethod

from core.environment.trade_state import AgentState
from lib.utils.stm import CueMemoryMatcher


class AgentStateMemoryMatcher(CueMemoryMatcher, ABC):

	@abstractmethod
	def is_match(self, cue: AgentState, memory: AgentState) -> bool:
		pass
