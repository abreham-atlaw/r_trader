from core.agent.agents.montecarlo_agent.stm.asmm import AgentStateMemoryMatcher
from core.agent.agents.montecarlo_agent.stm.msmm import MarketStateMemoryMatcher
from core.environment.trade_state import TradeState
from lib.utils.stm import CueMemoryMatcher


class TradeStateMemoryMatcher(CueMemoryMatcher):

	def __init__(
			self,
			agent_state_matcher: AgentStateMemoryMatcher,
			market_state_matcher: MarketStateMemoryMatcher
	):
		super().__init__()
		self.__agent_state_matcher = agent_state_matcher
		self.__market_state_matcher = market_state_matcher

	def is_match(self, cue: TradeState, memory: TradeState) -> bool:
		return (
			self.__agent_state_matcher.is_match(cue.get_agent_state(), memory.get_agent_state())
			and self.__market_state_matcher.is_match(cue.get_market_state(), memory.get_market_state())
		)
