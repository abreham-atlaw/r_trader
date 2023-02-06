import typing

import numpy as np

from lib.rl.environment import Environment


class TicTacToeEnvironment(Environment):

	class Reward:
		WIN = 10
		LOSS = -10
		DRAW = 0
		TIME = -1

	def __init__(self):
		super(TicTacToeEnvironment, self).__init__(True)
		self.state = None
		self.__current_player = 1

	@staticmethod
	def get_winner(state) -> int:
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
			reward = TicTacToeEnvironment.Reward.DRAW
		elif winner == 1:
			reward = TicTacToeEnvironment.Reward.WIN
		else:
			reward = TicTacToeEnvironment.Reward.LOSS

		reward += TicTacToeEnvironment.Reward.TIME

		return reward

	def perform_action(self, action):
		self.state[action[0], action[1]] = self.__current_player
		self.__current_player = 3 - self.__current_player
		if self.__current_player == 2 and not self.is_episode_over(self.get_state()):
			self.render()
			action_raw = int(input("Enter your move(1-9): ")) - 1
			action = (action_raw // 3, action_raw % 3)
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

	def get_valid_actions(self, state=None) -> typing.List:
		if state is None:
			state = self.get_state()
		valid_actions = []
		for i in range(3):
			for j in range(3):
				if state[i, j] == 0:
					valid_actions.append((i, j))
		return valid_actions

	def is_action_valid(self, action, state) -> bool:
		return action in self.get_valid_actions(state)

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
