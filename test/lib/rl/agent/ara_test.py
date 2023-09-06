import typing

import unittest
import time

from test.lib.rl.environment.environments.chess import ChessEnvironment, ChessGame
from .agents.chess import ChessMonteCarloAgent, ChessModelBasedAgent, ChessActionChoiceAgent, ChessStockfishModelBasedAgent


class ChessAgent(ChessMonteCarloAgent, ChessStockfishModelBasedAgent, ChessModelBasedAgent, ChessActionChoiceAgent):

	def perform_timestep(self):
		while self._get_environment().get_state().get_current_player() != self._get_environment().get_state().get_player_side():
			time.sleep(1)
		if self._get_environment().is_episode_over():
			return
		super().perform_timestep()


class ActionRecommendationAgentTest(unittest.TestCase):

	def test_functionality(self):

		agent0 = ChessAgent(
			explore_exploit_tradeoff=1.0,
			# update_batch_size=1,
			# clear_update_batch=False,
			use_stm=False,
			step_time=60
		)
		agent1 = ChessAgent(
			explore_exploit_tradeoff=1.0,
			# update_batch_size=1,
			# clear_update_batch=False,
			use_stm=False,
			step_time=60
		)

		game = ChessGame(agent0, agent1)
		game.start()
