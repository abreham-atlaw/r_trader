from typing import *

import unittest
from unittest import mock

import numpy as np

from copy import deepcopy

from core.environment.live_environment import LiveEnvironment, MarketState, AgentState, TradeState, TraderAction
from core.agent.agents import TraderMonteCarloAgent, TraderAgent
from temp import stats


class TraderAgentTest(unittest.TestCase):

	def setUp(self):
		self.agent = TraderAgent()
		self.environment = mock.Mock()
		self.agent.set_environment(self.environment)

	def test_prediction_to_transition_probability(self):
		market_state = MarketState(
			currencies=["USD", "EUR", "AUD"],
			tradable_pairs=[
				("USD", "EUR"),
				("AUD", "EUR"),
				("USD", "AUD")
			],
			memory_len=5
		)
		market_state.update_state_of("USD", "EUR", np.arange(5, 10))
		market_state.update_state_of("AUD", "EUR", np.arange(1, 5))
		market_state.update_state_of("USD", "AUD", np.arange(7, 12))
		initial_state = mock.Mock()
		initial_state.get_market_state.get_return = market_state

		final_market_state = deepcopy(market_state)
		final_market_state.update_state_of("AUD", "EUR", np.array([0.5]))
		final_state = mock.Mock()
		final_state.get_market_state.get_return = final_market_state

		output = 0.7

		result = self.agent._prediction_to_transition_probability(
			initial_state,
			np.array([0.7]),
			final_state
		)

		self.assertEqual(result, 1-output)

	def test_get_train_output(self):
		market_state = MarketState(
			currencies=["USD", "EUR", "AUD"],
			tradable_pairs=[
				("USD", "EUR"),
				("AUD", "EUR"),
				("USD", "AUD")
			],
			memory_len=5
		)
		market_state.update_state_of("USD", "EUR", np.arange(5, 10))
		market_state.update_state_of("AUD", "EUR", np.arange(1, 5))
		market_state.update_state_of("USD", "AUD", np.arange(7, 12))
		initial_state = mock.Mock()
		initial_state.get_market_state.get_return = market_state

		final_market_state = deepcopy(market_state)
		final_market_state.update_state_of("AUD", "EUR", np.array([0.5]))
		final_state = mock.Mock()
		final_state.get_market_state.get_return = final_market_state

		result = self.agent._get_train_output(
			initial_state,
			None,
			final_state
		)

		self.assertEqual(result, np.array([0]))

	def test_get_possible_states(self):
		market_state = MarketState(
			currencies=["USD", "EUR", "AUD"],
			tradable_pairs=[
				("USD", "EUR"),
				("AUD", "EUR"),
				("USD", "AUD")
			],
			memory_len=5
		)
		market_state.update_state_of("USD", "EUR", np.arange(5, 10))
		market_state.update_state_of("AUD", "EUR", np.arange(1, 6))
		market_state.update_state_of("USD", "AUD", np.arange(7, 12))

		initial_balance = 100
		agent_state = AgentState(initial_balance, market_state)
		agent_state.open_trade(
			TraderAction("USD", "EUR", TraderAction.Action.SELL, margin_used=40),
			current_value=8
		)

		state = TradeState(market_state, agent_state)

		result = self.agent._get_possible_states(
			state,
			TraderAction(
				"USD",
				"EUR",
				TraderAction.Action.CLOSE
			)
		)

		self.assertEqual(
			len(result),
			3*2
		)

		for p_state in result:
			self.assertGreater(
				p_state.agent_state.get_balance(),
				initial_balance
			)
			self.assertEqual(
				len(p_state.agent_state.get_open_trades()),
				0
			)

	def test_perform_timestep(self):
		environment = LiveEnvironment()
		environment.start()
		self.agent.set_environment(environment)
		self.agent.perform_timestep()

	def test_loop(self):
		environment = LiveEnvironment()
		environment.start()
		self.agent.set_environment(environment)
		self.agent.loop()

	def test_resume_mca(self):
		environment = LiveEnvironment()
		environment.start()

		agent = TraderAgent()
		agent.set_environment(environment)

		node, repo = stats.load_and_draw_graph("/home/abrehamatlaw/Downloads/Compressed/results/graph_dumps/1723586895.457289")
		state = repo.retrieve(node.id)

		agent._monte_carlo_tree_search(state)

		x = 1
