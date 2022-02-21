from typing import *

import unittest

import os
import sys
from copy import deepcopy

from lib.rl.environment import Environment
from lib.rl.agent import MarkovAgent


class AgentCacheTest(unittest.TestCase):
	ACTIONS = [1, 2, 3, 4]
	STATES = [15, 6, 2]
	VALUES = [
		[1, 2, 3, 4],
		[5, 6, 7, 8],
		[9, 0, 1, 2],
	]

	def setUp(self) -> None:
		self.cacher = MarkovAgent.Cacher()

	def test_valid_cache(self):

		for i, state in enumerate(AgentCacheTest.STATES):
			for j, action in enumerate(AgentCacheTest.ACTIONS):
				self.cacher.cache(
					state, action, AgentCacheTest.VALUES[i][j], 10
				)

		for i, state in enumerate(AgentCacheTest.STATES):
			for j, action in enumerate(AgentCacheTest.ACTIONS):
				value = self.cacher.get_cached(state, action, 3)
				self.assertEqual(AgentCacheTest.VALUES[i][j], value)

	def test_invalid_cache(self):

		value = self.cacher.get_cached(4, 4, 2)
		self.assertIsNone(value)

		self.cacher.cache(3, 4, 4, 5)
		value = self.cacher.get_cached(3, 4, 6)
		self.assertIsNone(value)


class AgentTest(unittest.TestCase):
	class HanoiTowerEnvironment(Environment):

		SUCCESS_REWARD = 200
		TIME_PENALITY = -1

		def __init__(self, disks: int, target_tower: int, towers: int = 3):
			super().__init__()
			self.disks_num = disks
			self.towers_num = towers
			self.state: Union[List[List[int]], None] = None
			self.initial_tower = 0
			self.target_tower = target_tower

		def is_game_over(self, state: List[List[int]]) -> bool:
			if len(state[self.target_tower]) == self.disks_num:
				return True
			return False

		def get_reward(self, state: List[List[int]] = None) -> float:
			if state is None:
				state = self.get_state()
			if self.is_game_over(state):
				return AgentTest.HanoiTowerEnvironment.SUCCESS_REWARD
			return AgentTest.HanoiTowerEnvironment.TIME_PENALITY

		def get_state(self) -> List[List[int]]:
			if self.state is None:
				raise Exception("Environment Not Initialized.")
			return self.state

		def perform_action(self, action: Tuple[int, int]):  # Action consists of source tower and destination tower.
			self.get_state()[action[1]].append(self.get_state()[action[0]].pop())

		def render(self):
			for i in range(self.disks_num):
				for tower in self.get_state():
					if len(tower) < (self.disks_num - i):
						layer = "|"
					else:
						layer = "_" * tower[self.disks_num - (i + 1)] * 3
					print(f"{layer:^25s}", end="")
				print()

		def update_ui(self):
			os.system("clear")
			self.render()

		def check_is_running(self) -> bool:
			return not self.is_episode_over()

		def get_valid_actions(self, state=None) -> List[Tuple[int, int]]:
			if state is None:
				state = self.get_state()

			actions = []

			for i, source_tower in enumerate(state):
				for j, destination_tower in enumerate(state):
					if i == j or len(source_tower) == 0:
						continue
					if len(destination_tower) == 0 or source_tower[-1] < destination_tower[-1]:
						actions.append((i, j))

			return actions

		def is_episode_over(self, state=None) -> bool:
			if state is None:
				state = self.get_state()
			return self.is_game_over(state)

		def reset(self):
			self.state = None
			self._initialize()

		def _initialize(self):
			self.state = [list(range(self.disks_num, 0, -1))] + [[] for i in range(self.towers_num - 1)]
			super()._initialize()

	class HanoiTowerAgent(MarkovAgent):

		def _get_expected_transition_probability(self, initial_state: List[List[int]], action: Tuple[int, int],
												final_state: List[List[int]]) -> float:
			return 1.0

		def _update_transition_probability(self, initial_state: List[List[int]], action: Tuple[int, int],
											final_state: List[List[int]]):
			pass

		def _get_expected_instant_reward(self, state: List[List[int]]) -> float:
			return self._get_environment().get_reward(state)

		def _get_possible_states(self, state: List[List[int]], action: Tuple[int, int]) -> List[object]:
			state = deepcopy(state)
			state[action[1]].append(state[action[0]].pop())
			return [state]

	def setUp(self):
		sys.setrecursionlimit(
			sys.getrecursionlimit() * 3
		)

	def test_hanoi_agent_in_mocked_environment(self):
		agent = AgentTest.HanoiTowerAgent(depth=10, discount=1, explore_exploit_tradeoff=1)
		environment = AgentTest.HanoiTowerEnvironment(3, 2)
		environment.start()
		agent.set_environment(environment)
		agent.perform_episode()


if __name__ == "__main__":
	unittest.main()
