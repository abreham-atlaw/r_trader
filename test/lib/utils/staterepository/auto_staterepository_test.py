import unittest

from core.environment.trade_state import MarketState, TradeState, AgentState
from lib.utils.staterepository import AutoStateRepository


class AutoStateRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.SIZE = 5
		self.repository = AutoStateRepository(memory_size=self.SIZE)
		market_state = MarketState(
			currencies=["AUD", "USD"],
			memory_len=1024,
			tradable_pairs=[("AUD", "USD")]
		)
		self.STATES = [
			TradeState(
				market_state=market_state,
				agent_state=AgentState(
					balance=i,
					market_state=market_state
				)
			)
			for i in range(10)
		]

	def test_functionality(self):

		for i, state in enumerate(self.STATES):
			print(f"Storing {i}..")
			self.repository.store(f"id-{i}-{i}", state)

		for i, state in enumerate(self.STATES):
			print(f"Getting {i}...")
			self.assertEqual(state.agent_state.get_balance(), self.repository.retrieve(f"id-{i}-{i}").agent_state.get_balance())

		self.repository.destroy()

