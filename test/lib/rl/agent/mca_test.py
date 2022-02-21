from typing import *

import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses

from datetime import datetime
from copy import deepcopy

from lib.rl.agent import MonteCarloAgent, DNNTransitionAgent
from lib.rl.environment import Environment


class MonteCarloAgentTest(unittest.TestCase):
	class TicTacToeEnvironment(Environment):

		class Reward:
			WIN = 5
			LOSS = -5
			DRAW = 0
			TIME = -1

		def __init__(self):
			super(MonteCarloAgentTest.TicTacToeEnvironment, self).__init__(True)
			self.state = None
			self.__current_player = 1

		def get_winner(self, state) -> int:
			for i in range(3):
				for row in [state[i, :], state[:, i]]:
					if row[0] == row[1] == row[2] != 0:
						return row[0]
			if state[0, 0] == state[1, 1] == state[2, 2] != 0:
				return state[1, 1]
			if state[0, 2] == state[1, 1] == state[2, 0] != 0:
				return state[1, 1]

			return 0

		def get_reward(self, state=None) -> float:
			if state is None:
				state = self.get_state()
			winner = self.get_winner(state)
			if winner == 0:
				reward = MonteCarloAgentTest.TicTacToeEnvironment.Reward.DRAW
			elif winner == 1:
				reward = MonteCarloAgentTest.TicTacToeEnvironment.Reward.WIN
			else:
				reward = MonteCarloAgentTest.TicTacToeEnvironment.Reward.LOSS

			# reward += MonteCarloAgentTest.TicTacToeEnvironment.Reward.TIME

			return reward

		def perform_action(self, action):
			self.state[action[0], action[1]] = self.__current_player
			self.__current_player = 3 - self.__current_player
			if self.__current_player == 2 and not self.is_episode_over(self.get_state()):
				self.render()
				action_raw = int(input("Enter your move(1-9): ")) - 1
				action = (action_raw//3, action_raw%3)
				self.perform_action(action)

		def render(self):
			print("\n" * 2)
			for row in self.get_state():
				for cell in row:
					if cell == 0:
						char = " "
					elif cell == 1:
						char = "O"
					else:
						char = "X"
					print(f"| {char} |", end="")
				print()

		def update_ui(self):
			self.render()

		def check_is_running(self) -> bool:
			return True

		def get_valid_actions(self, state=None) -> List:
			if state is None:
				state = self.get_state()
			valid_actions = []
			for i in range(3):
				for j in range(3):
					if state[i, j] == 0:
						valid_actions.append((i, j))
			return valid_actions

		def get_state(self):
			return self.state

		def is_episode_over(self, state=None) -> bool:
			if state is None:
				state = self.get_state()

			return self.get_winner(state) != 0 or 0 not in state

		def _initialize(self):
			self.state = np.zeros((3, 3))
			self.__current_player = 1
			super()._initialize()

	class TicTacToeAgent(MonteCarloAgent, DNNTransitionAgent):

		def __init__(self, *args, step_time=120, **kwargs):
			super(MonteCarloAgentTest.TicTacToeAgent, self).__init__(*args, **kwargs)
			self.__step_time = step_time
			self.set_transition_model(self.create_transition_model())

		def create_transition_model(self) -> keras.Model:
			model = keras.Sequential()
			model.add(layers.InputLayer((9,)))
			model.add(layers.Dense(27))
			model.add(layers.Dense(10))
			model.add(layers.Dense(9, activation=activations.softmax))

			model.compile(optimizer=optimizers.adam_v2.Adam(), loss=losses.categorical_crossentropy)

			return model

		def _init_resources(self) -> object:
			start_time = datetime.now()
			return start_time

		def _has_resource(self, start_time) -> bool:
			return (datetime.now() - start_time).seconds < self.__step_time

		def _state_action_to_model_input(self, state, action, final_state) -> np.ndarray:
			mid_state: np.ndarray = deepcopy(state)
			mid_state[action[0], action[1]] = 1

			return mid_state.reshape(-1)

		def __get_output_index(self, initial_state, final_state) -> int:
			for i in range(3):
				for j in range(3):
					if final_state[i, j] == 2 and initial_state[i, j] == 0:
						return (i * 3) + j

		def _prediction_to_transition_probability(self, initial_state, output: np.ndarray, final_state) -> float:

			if self._get_environment().get_winner(final_state) == 1 or 0 not in final_state:
				return 1.0

			output = tf.reshape(output, (-1,)).numpy()
			index = self.__get_output_index(initial_state, final_state)

			for i in range(3):
				for j in range(3):
					action_index = (i * 3) + j
					if final_state[i, j] != 0 and action_index != index:
						output[action_index] = 0

			output /= output.sum()

			return output[index]

		def _get_train_output(self, initial_state, action, final_state) -> np.ndarray:
			index = self.__get_output_index(initial_state, final_state)

			output = np.zeros((9,))
			output[index] = 1
			return output

		def _get_expected_instant_reward(self, state) -> float:
			return self._get_environment().get_reward(state)

		def _get_possible_states(self, state, action) -> List[object]:

			mid_state = deepcopy(state)
			mid_state[action[0], action[1]] = 1

			if self._get_environment().is_episode_over(mid_state):
				return [mid_state]

			possible_states = []
			for i in range(3):
				for j in range(3):
					if mid_state[i, j] == 0:
						new_state = deepcopy(mid_state)
						new_state[i, j] = 2
						possible_states.append(new_state)

			return possible_states

	def setUp(self) -> None:
		self.environment = MonteCarloAgentTest.TicTacToeEnvironment()
		self.environment.start()
		self.agent = MonteCarloAgentTest.TicTacToeAgent(
			step_time=10,
			episodic=True,
			depth=10,
			explore_exploit_tradeoff=1,
			discount=0.3
		)
		self.agent.set_environment(self.environment)

	def test_functionality(self):
		self.agent.loop()

	def test_best_option(self):

		for i in range(10):
			self.environment.state = np.array([
				[2, 0, 1],
				[0, 1, 0],
				[2, 0, 0]
			])
			action = self.agent._policy(self.environment.state)
			self.assertEqual((1, 0), action)
