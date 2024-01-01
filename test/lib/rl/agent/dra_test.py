import time
import unittest

from test.lib.rl.agent.agents.chess import ChessModelBasedAgent, ChessActionChoiceAgent
from test.lib.rl.agent.agents.dra import ChessDeepReinforcementAgent
from test.lib.rl.environment.environments.chess import ChessGame


class ChessAgent(ChessDeepReinforcementAgent, ChessActionChoiceAgent):

	def perform_timestep(self):
		while self._get_environment().get_state().get_current_player() != self._get_environment().get_state().get_player_side():
			time.sleep(1)
		if self._get_environment().is_episode_over():
			return
		super().perform_timestep()


class DeepReinforcementAgentTest(unittest.TestCase):

	def test_functionality(self):

		agent0 = ChessAgent(
			train=True,
			save_path="/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl/chess/agent_0",
			explore_exploit_tradeoff=1.0,
		)
		agent1 = ChessAgent(
			train=True,
			save_path="/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl/chess/agent_1",
			explore_exploit_tradeoff=1.0,
		)

		game = ChessGame(agent0, agent1)
		game.start()
