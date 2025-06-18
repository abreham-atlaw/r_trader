from core import Config
from core.agent.agents.montecarlo_agent.stm.msmm import MarketStateMemoryMatcher, BoundMarketStateMemoryMatcher
from test.core.agent.monte_carlo_agent.stm.abstract_msmm_test import AbstractMarketStateMemoryMatcherTest


class BoundMarketStateMemoryMatcherTest(AbstractMarketStateMemoryMatcherTest):

	def init_matcher(self) -> MarketStateMemoryMatcher:
		return BoundMarketStateMemoryMatcher(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)

	def test_valid(self):
		super().test_valid()

	def test_invalid(self):
		super().test_invalid()
