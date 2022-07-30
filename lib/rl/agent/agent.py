from typing import *
from abc import ABC, abstractmethod

import numpy as np

import os
import json
import random


from lib.rl.environment import Environment


CONFIGS_FILE_NAME = "configs.txt"


class Agent(ABC):

	def __init__(self, episodic = False, explore_exploit_tradeoff: float = 0.3, update_agent=True):
		self.__environment: Union[Environment, None] = None
		self._is_episodic = episodic
		self._explore_exploit_tradeoff = explore_exploit_tradeoff
		self._update_agent = update_agent

	def _get_environment(self) -> Environment:
		if self.__environment is None:
			raise Exception("Environment Not Set.")
		return self.__environment

	def set_environment(self, environment: Environment):
		self.__environment = environment

	def _get_available_actions(self, state) -> List[object]:
		return self._get_environment().get_valid_actions(state)

	def _is_episode_over(self, state) -> bool:
		if self._is_episodic:
			return self._get_environment().is_episode_over(state)
		return False

	@abstractmethod
	def _get_state_action_value(self, state, action, **kwargs) -> float:
		pass

	@abstractmethod
	def _update_state_action_value(self, initial_state, action, final_state, value):
		pass

	def _get_optimal_action(self, state, **kwargs):
		valid_actions = self._get_available_actions(state)

		values = [self._get_state_action_value(state, action, **kwargs) for action in valid_actions]

		optimal_action = valid_actions[values.index(max(values))]
		return optimal_action

	def _policy(self, state, **kwargs):
		return self._get_optimal_action(state, **kwargs)

	def _exploit(self, state):
		return self._policy(state)

	def _explore(self, state):
		actions = self._get_available_actions(state)
		return random.choice(actions)

	def _get_action(self, state):
		method = np.random.choice(
			[self._exploit, self._explore],
			1,
			p=[self._explore_exploit_tradeoff, 1-self._explore_exploit_tradeoff]
		)[0]
		return method(state)

	def perform_timestep(self):
		state = self._get_environment().get_state()
		action = self._get_action(state)
		value = self._get_environment().do(action)
		if self._update_agent:
			self._update_state_action_value(state, action, self._get_environment().get_state(), value)

	def perform_episode(self):
		while not self._get_environment().is_episode_over():
			self.perform_timestep()

	def loop(self):
		if self._is_episodic:
			while True:
				self.perform_episode()
				self._get_environment().reset()
		else:
			while True:
				self.perform_timestep()

	def get_configs(self) -> Dict:
		return {
			"explore_exploit_tradeoff": self._explore_exploit_tradeoff,
			"discount": self._discount_factor,
			"depth": self._depth,
			"episodic": self._is_episodic
		}

	def save(self, location):
		configs = self.get_configs()
		Agent.save_configs(configs, location)

	@staticmethod
	def save_configs(configs: Dict, location):
		if os.path.isdir(location):
			os.mkdir(location)

		with open(os.path.join(location, CONFIGS_FILE_NAME), "w") as output:
			json.dump(configs, output)
			output.close()

	@staticmethod
	def load_configs(location) -> Dict:

		with open(os.path.join(location, CONFIGS_FILE_NAME)) as configs_file:
			data = json.load(configs_file)
			return data

	@classmethod
	def load_agent(cls, location):
		configs = cls.load_configs(location)
		return cls(**configs)
