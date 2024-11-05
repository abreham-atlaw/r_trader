import typing

import numpy as np

import unittest

from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer
from lib.utils.staterepository import DictStateRepository, SectionalDictStateRepository, StateRepository
from lib.rl.agent.mca import MonteCarloAgent, Node
from core.environment.trade_state import TradeState, MarketState, AgentState
from core.agent.agents.montecarlo_agent.stm import TraderNodeMemoryMatcher, TraderNodeShortTermMemory, TraderNodeMemory
from temp import stats


class TraderSTMTest(unittest.TestCase):

	@staticmethod
	def __setup_nodes_and_repo():
		previous_node, previous_repo = stats.load_node_repo(
			"/home/abrehamatlaw/Downloads/Compressed/results_12/graph_dumps/1729670464.530439")
		predicted_node = previous_node.children[2].children[3]
		incorrect_node = previous_node.children[2].children[2]
		print(
			f"Previous Node:\nID: {previous_node.id}, Price: {previous_repo.retrieve(previous_node.id).market_state.get_current_price('AUD', 'USD')}")
		print(
			f"Similiar Node:\nID: {predicted_node.id}, Price: {previous_repo.retrieve(predicted_node.id).market_state.get_current_price('AUD', 'USD')}")
		print(
			f"Incorrect Node:\nID: {incorrect_node.id}, Price: {previous_repo.retrieve(incorrect_node.id).market_state.get_current_price('AUD', 'USD')}"
		)

		next_node, next_repo = stats.load_node_repo(
			"/home/abrehamatlaw/Downloads/Compressed/results_12/graph_dumps/1729670646.049332")
		print(
			f"Next Node:\nID: {next_node.id}, Price: {next_repo.retrieve(next_node.id).market_state.get_current_price('AUD', 'USD')}")

		repo = SectionalDictStateRepository(2, 15, serializer=TraderNodeSerializer())
		repo.store(predicted_node.id, previous_repo.retrieve(predicted_node.id))
		repo.store(incorrect_node.id, previous_repo.retrieve(incorrect_node.id))
		repo.store(next_node.id, next_repo.retrieve(next_node.id))

		return predicted_node, incorrect_node, next_node, repo

	def setUp(self) -> None:

		self.memory_size = 10
		self.similar_node, self.incorrect_node, self.target_node, self.repo = self.__setup_nodes_and_repo()
		self.threshold = 1e-4
		self.matcher = TraderNodeMemoryMatcher(
			threshold=self.threshold,
			repository=self.repo,
			use_ma_smoothng=False,
			mean_error=False
		)
		self.memory = TraderNodeShortTermMemory(
			matcher=self.matcher,
			size=self.memory_size
		)
		for node in [self.similar_node, self.incorrect_node]:
			self.memory.memorize(node)

	def test_successful_match(self):
		self.assertTrue(
			self.matcher.is_match(
				self.similar_node,
				TraderNodeMemory(
					node=self.target_node
				)
			)
		)

	def test_unsuccessful_match(self):
		self.assertFalse(
			self.matcher.is_match(
				self.incorrect_node,
				TraderNodeMemory(
					node=self.target_node
				)
			)
		)

	def test_successful_recall(self):
		self.assertIs(
			self.memory.recall(
				self.target_node
			),
			self.similar_node
		)
