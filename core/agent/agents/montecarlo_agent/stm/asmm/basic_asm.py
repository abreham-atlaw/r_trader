import numpy as np

from core.environment.trade_state import AgentState
from .asmm import AgentStateMemoryMatcher


class BasicAgentStateMemoryMatcher(AgentStateMemoryMatcher):

	def __init__(self, *args, balance_tolerance: float = 0.01, **kwargs):
		super().__init__(*args, **kwargs)
		self.__balance_tolerance = balance_tolerance

	@staticmethod
	def __compare_trades(trade0: AgentState.OpenTrade, trade1: AgentState.OpenTrade) -> bool:
		return trade0.get_trade() == trade1.get_trade()

	def is_match(self, cue: AgentState, memory: AgentState) -> bool:
		return \
				np.isclose(cue.get_balance(), memory.get_balance(), atol=self.__balance_tolerance) and \
				len(cue.get_open_trades()) == len(memory.get_open_trades()) and \
				np.all([self.__compare_trades(
					trade0,
					trade1
				) for trade0, trade1 in zip(*[
					sorted(state.get_open_trades(), key=lambda t: t.get_enter_value())
					for state in [cue, memory]
				]
											)
				])
