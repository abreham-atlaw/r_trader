import typing
from typing import List

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models

import unittest

from lib.rl.agent.drl import DeepReinforcementAgent
from test.lib.rl.environment.environments.chess import ChessGame
from .agents.chess import ChessGameAgent, ChessActionChoiceAgent, ChessDeepReinforcementAgent


class ChessAgent(ChessGameAgent, ChessDeepReinforcementAgent, ChessActionChoiceAgent):
	pass


class DeepReinforcementAgentTest(unittest.TestCase):

	class TicTacToeAgent(DeepReinforcementAgent):

		def _prepare_dra_input(self, state: np.ndarray, action: typing.Tuple[int, int]) -> np.ndarray:
			return np.concatenate((state.flatten(), action))

		def _prepare_dra_output(self, state: typing.Any, action: typing.Any, output: np.ndarray) -> float:
			return float(output)

		def _prepare_train_value(self, state: typing.Any, action: typing.Any, value: float) -> np.ndarray:
			return np.array(value)

		def _fit_model(self, model: models.Model, X: np.ndarray, y: np.ndarray):
			model.fit(X, y, epochs=1000)

		def _init_dra_model(self) -> models.Model:
			model = models.Sequential()
			model.add(layers.Dense(100, input_shape=(11,), activation="relu"))
			model.add(layers.Dense(50, activation="relu"))
			model.add(layers.Dense(10, activation="relu"))
			model.add(layers.Dense(1))
			model.compile(optimizer="adam", loss="mse")
			return model

		def _generate_actions(self, state) -> List[object]:
			return self._get_environment().get_valid_actions()

	def test_functionality(self):
		agent0 = ChessAgent(explore_exploit_tradeoff=1.0, batch_size=1)
		agent1 = ChessAgent(explore_exploit_tradeoff=1.0, batch_size=1)

		game = ChessGame(agent0, agent1)
		game.start()
