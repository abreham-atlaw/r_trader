from typing import *

import numpy as np

import unittest

from lib.utils.staterepository import DictStateRepository
from lib.rl.agent.mca import MonteCarloAgent
from core.environment.trade_state import TradeState, MarketState, AgentState
from core.agent.stm import TraderNodeMemoryMatcher, TraderNodeShortTermMemory, TraderNodeMemory


class TraderSTMTest(unittest.TestCase):

	ORIGINAL_NODE = MonteCarloAgent.Node(None, None, 0, weight=0.8, instant_value=0.45)
	ORIGINAL_MARKET_STATE = MarketState(
		currencies=["AUD", "GBP", "USD"],
		state=np.arange(3*3*10).reshape((3, 3, 10))
	)
	ORIGINAL_AGENT_STATE = AgentState(
		balance=100,
		market_state=ORIGINAL_MARKET_STATE
	)
	ORIGINAL_STATE = TradeState(
		market_state=ORIGINAL_MARKET_STATE,
		agent_state=ORIGINAL_AGENT_STATE
	)

	SIMILAR_NODE = MonteCarloAgent.Node(None, None, 0, weight=0.8, instant_value=0.45)
	SIMILAR_MARKET_STATE = MarketState(
		currencies=["AUD", "GBP", "USD"],
		state=np.arange(3*3*10).reshape((3, 3, 10)) * 0.99
	)
	SIMILAR_AGENT_STATE = AgentState(
		balance=100,
		market_state=SIMILAR_MARKET_STATE
	)
	SIMILAR_STATE = TradeState(
		market_state=SIMILAR_MARKET_STATE,
		agent_state=SIMILAR_AGENT_STATE
	)

	DIFFERENT_NODE = MonteCarloAgent.Node(None, None, 0, weight=0.8, instant_value=0.45)
	DIFFERENT_MARKET_STATE = MarketState(
		currencies=["AUD", "GBP", "USD"],
		state=np.arange(3 * 3 * 10).reshape((3, 3, 10)) * 0.5
	)
	DIFFERENT_AGENT_STATE = AgentState(
		balance=100,
		market_state=DIFFERENT_MARKET_STATE
	)
	DIFFERENT_STATE = TradeState(
		market_state=DIFFERENT_MARKET_STATE,
		agent_state=DIFFERENT_AGENT_STATE
	)

	def setUp(self) -> None:
		repository = DictStateRepository()
		for node, state in zip(
				[
					TraderSTMTest.ORIGINAL_NODE,
					TraderSTMTest.SIMILAR_NODE,
					TraderSTMTest.DIFFERENT_NODE
				],
				[
					TraderSTMTest.ORIGINAL_STATE,
					TraderSTMTest.SIMILAR_STATE,
					TraderSTMTest.DIFFERENT_STATE
				]
		):
			repository.store(node.id, state)
		self.matcher = TraderNodeMemoryMatcher(0.02, repository=repository, average_window=10)
		self.stm = TraderNodeShortTermMemory(5, 0.02, average_window=10)
		self.stm.set_matcher(self.matcher)
		self.stm.memorize(TraderSTMTest.ORIGINAL_NODE)

	def test_successful_match(self):
		self.assertTrue(
			self.matcher.is_match(
				TraderSTMTest.SIMILAR_NODE,
				TraderNodeMemory(
					node=TraderSTMTest.ORIGINAL_NODE
				)
			)
		)

	def test_unsuccessful_match(self):
		self.assertFalse(
			self.matcher.is_match(
				TraderSTMTest.DIFFERENT_NODE,
				TraderNodeMemory(
					node=TraderSTMTest.ORIGINAL_NODE
				)
			)
		)

	def test_successful_recall(self):
		self.assertIs(
			self.stm.recall(
				TraderSTMTest.SIMILAR_NODE
			),
			TraderSTMTest.ORIGINAL_NODE
		)

	def test_unsuccessful_recall(self):
		self.assertIsNone(
			self.stm.recall(
				TraderSTMTest.DIFFERENT_NODE
			)
		)

