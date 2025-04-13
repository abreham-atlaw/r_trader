import unittest

from core.environment.trade_state import TradeState, MarketState, AgentState
from lib.utils.staterepository import PickleStateRepository


class PickleStateRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.repository = PickleStateRepository()
		market_state = MarketState(
			currencies=["AUD", "USD"],
			memory_len=1024,
			tradable_pairs=[("AUD", "USD")]
		)
		self.state = TradeState(
			market_state=market_state,
			agent_state=AgentState(
				balance=98.0,
				market_state=market_state
			)
		)
		self.key = "test_key"

	def test_functionality(self):
		self.repository.store(self.key, self.state)

		state: TradeState = self.repository.retrieve(self.key)
		self.assertEqual(state.market_state.get_currencies(), self.state.market_state.get_currencies())
