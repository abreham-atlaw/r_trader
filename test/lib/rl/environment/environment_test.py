from typing import *

import unittest

import os
from copy import deepcopy

from lib.rl.environment import Environment, ActionNotValidException


class EnvironmentTest(unittest.TestCase):

	
	class HanoiTowerEnvironment(Environment):

		SUCCESS_REWARD = 20
		TIME_PENALITY = -1

		def __init__(self, disks: int, towers: int=3):
			self.disks_num = disks
			self.towers_num = towers
			self.state: Union[List[List[int]], None] = None
			self.initial_tower = 0

		def is_game_over(self, state: List[List[int]]) -> bool:
			if len(state[self.initial_tower]) == self.disks_num:
				return True
			return False

		def get_reward(self, state: List[List[int]]=None) -> float:
			if state is None:
				state = self.get_state()
			if self.is_game_over(state):
				return EnvironmentTest.HanoiTowerEnvironment.SUCCESS_REWARD
			return EnvironmentTest.HanoiTowerEnvironment.TIME_PENALITY
		
		def get_state(self) -> List[List[int]]:
			if self.state is None:
				raise Exception("Environment Not Initialized.")
			return self.state
		
		def perform_action(self, action: Tuple[int, int]): # Action consists of source tower and destination tower.
			self.get_state()[action[1]].append(self.get_state()[action[0]].pop())
		
		def render(self):
			for i in range(self.disks_num):
				for tower in self.get_state():
					if len(tower) < (self.disks_num - i):
						layer = "|"
					else:
						layer = "_" * tower[self.disks_num - (i+1)] * 3
					print(f"{layer:^25s}", end="")
				print()
		
		def update_ui(self):
			os.system("clear")
			self.render()
		
		def check_is_running(self) -> bool:
			return not self.is_episode_over()
		
		def get_valid_actions(self, state = None) -> List[Tuple[int, int]]:
			if state == None:
				state = self.get_state()
			
			actions = []

			for i, source_tower in enumerate(state):
				for j, destination_tower in enumerate(state):
					if i == j or len(source_tower) == 0:
						continue
					if len(destination_tower) == 0 or source_tower[-1] < destination_tower[-1]:
						actions.append((i, j))
			
			return actions
		
		def is_episode_over(self) -> bool:
			return self.is_game_over(self.get_state())
		
		def reset(self):
			self.state = None
			self._initialize()

		def _initialize(self):
			self.state = [list(range(self.disks_num, 0, -1))] + [[] for i in range(self.towers_num - 1)]
			super()._initialize()
		

	def setUp(self) -> None:
		self.environment = EnvironmentTest.HanoiTowerEnvironment(5)
		self.environment.start()
	
	def test_initialize(self):
		self.assertIsNotNone(self.environment.get_state())

	def test_do(self):
		initial_state = deepcopy(self.environment.get_state())
		self.environment.do((0, 1))
		self.assertNotEqual(initial_state, self.environment.get_state())
	
	def test_do_exception(self):
		error = False
		self.environment.do((0,2))
		self.assertNotIn((0,2), self.environment.get_valid_actions())
		
		try:
			self.environment.do((0,2))
		except ActionNotValidException:
			error = True
		
		self.assertTrue(error)


if __name__ == "__main__":
	unittest.main()
