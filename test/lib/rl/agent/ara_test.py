import typing

import unittest
import time

from test.lib.rl.environment.environments.chess import ChessEnvironment, ChessGame
from .agents.chess import ChessActionRecommendationBalancerAgent, ChessMonteCarloAgent, ChessModelBasedAgent, ChessDNNTransitionAgent, ChessActionChoiceAgent


class ChessAgent(ChessActionRecommendationBalancerAgent, ChessMonteCarloAgent, ChessModelBasedAgent):

	def perform_timestep(self):
		while self._get_environment().get_state().get_current_player() != self._get_environment().get_state().get_player_side():
			time.sleep(1)
		if self._get_environment().is_episode_over():
			return
		super().perform_timestep()


class ActionRecommendationAgent(unittest.TestCase):

	def test_functionality(self):

		agent0 = ChessAgent(explore_exploit_tradeoff=1.0, num_actions=10, ara_tries=10, batch_size=1)
		agent1 = ChessAgent(explore_exploit_tradeoff=1.0, num_actions=10, ara_tries=10, batch_size=1)

		game = ChessGame(agent0, agent1)
		game.start()
