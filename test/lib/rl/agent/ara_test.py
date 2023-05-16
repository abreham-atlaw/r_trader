import typing

import unittest
import time

from test.lib.rl.environment.environments.chess import ChessEnvironment, ChessGame
from .agents.chess import ChessGameAgent, ChessActionRecommendationBalancerAgent, ChessMonteCarloAgent, ChessModelBasedAgent


class ChessAgent(ChessGameAgent, ChessActionRecommendationBalancerAgent, ChessMonteCarloAgent, ChessModelBasedAgent):
	pass


class ActionRecommendationAgent(unittest.TestCase):

	def test_functionality(self):

		agent0 = ChessAgent(explore_exploit_tradeoff=1.0, num_actions=10, ara_tries=10, batch_size=1)
		agent1 = ChessAgent(explore_exploit_tradeoff=1.0, num_actions=10, ara_tries=10, batch_size=1)

		game = ChessGame(agent0, agent1)
		game.start()
