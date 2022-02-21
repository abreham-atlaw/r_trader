from typing import *
from abc import ABC, abstractmethod

import numpy as np

import random
import os
import json

from lib.concurrency import Pool
from lib.rl.environment import Environment
from lib.utils.logger import Logger

EXPLORE_EXPLOIT_INSTANCES = 100
CONFIGS_FILE_NAME = "configs.txt"


class Agent(ABC):

	class Cacher:  # TODO: ADD DEPTH

		def __init__(self) -> None:
			self.__states = []
			self.__actions = []
			self.__cache = np.array([[[None, None]]])

		def __get_coordinates(self, state, action) -> Tuple[int, int]:
			return (self.__states.index(state), self.__actions.index(action))

		def __create_state(self, state):
			if len(self.__states) != 0:
				self.__cache = np.vstack([
					self.__cache,
					np.array([None]*len(self.__actions)*2).reshape((1, len(self.__actions), 2))
				])
				
			self.__states.append(state)
			return len(self.__states) - 1
		
		def __create_action(self, action):
			if len(self.__actions) != 0:
				self.__cache = np.hstack([
					self.__cache,
					np.array([None]*len(self.__states)*2).reshape((len(self.__states), 1, 2))
				])
			self.__actions.append(action)

		def cache(self, state, action, value: float, depth: int):
			if state not in self.__states:
				self.__create_state(state)
			if action not in self.__actions:
				self.__create_action(action)
			state_index, action_index = self.__get_coordinates(state, action)
			self.__cache[state_index, action_index] = [value, depth]
		
		def get_cached(self, state, action, min_depth) -> Union[float, None]:
			if state not in self.__states or action not in self.__actions:
				return None
			state_index, action_index = self.__get_coordinates(state, action)
			value, depth = self.__cache[state_index, action_index]
			if value is None or depth < min_depth:
				return None
			return value

		def clear(self):
			self.__states = []
			self.__actions = []
			self.__cache = np.array([[[None, None]]])

	def __init__(self, explore_exploit_tradeoff: float = 0.3, discount: float = 0.7, depth: int = None, episodic: bool = True, session_caching: bool = True):
		self._explore_exploit_tradeoff = explore_exploit_tradeoff
		self._discount_factor = discount
		self._is_episodic = episodic
		self._depth = depth
		self.__environment: Union[Environment, None] = None
		self.__session_caching = session_caching
		if session_caching:
			self.__session_cacher: Agent.Cacher = Agent.Cacher()
		if not self._is_episodic and self._depth is None:
			raise Exception("Non-Episodic Tasks can't have depth=None")

	def _get_environment(self) -> Environment:
		if self.__environment is None:
			raise Exception("Environment Not Set.")
		return self.__environment

	@Logger.logged_method
	def _get_available_actions(self, state) -> List[object]:
		return self._get_environment().get_valid_actions(state)

	@abstractmethod
	def _get_expected_transition_probability(self, initial_state, action, final_state) -> float:
		pass

	@abstractmethod
	def _update_transition_probability(self, initial_state, action, final_state):
		pass

	@abstractmethod
	def _get_expected_instant_reward(self, state) -> float:
		pass

	@abstractmethod
	def _get_possible_states(self, state, action) -> List[object]:
		pass

	def set_environment(self, environment: Environment):
		self.__environment = environment

	@Logger.logged_method
	def _is_episode_over(self, state) -> bool:
		if self._is_episodic:
			return self._get_environment().is_episode_over(state)
		return False  # Default value for non-episodic tasks

	@Logger.logged_method
	def _policy(self, state, depth):  # Override this function for non-value based policy
		return self._get_optimal_action(state, depth)

	def _on_concurrency_start(self):
		pass

	def _on_concurrency_end(self):
		pass

	def _get_state_action_value_async(self, state, action, depth):
		self._on_concurrency_start()
		return_value = self._get_state_action_value(state, action, depth)
		self._on_concurrency_end()
		return return_value

	@Logger.logged_method
	def _get_optimal_action(self, state, depth):
		valid_actions = self._get_available_actions(state)

		if Pool.is_main_process():
			pool = Pool()
			pool.map(self._get_state_action_value_async, [(state, action, depth) for action in valid_actions])
			expectations = pool.get_return()
		else:
			expectations = [self._get_state_action_value(state, action, depth) for action in valid_actions]

		optimal_action = valid_actions[expectations.index(max(expectations))]
		return optimal_action

	@Logger.logged_method
	def _get_state_action_value(self, state, action, depth):
		states = self._get_possible_states(state, action)
		value = 0

		if self.__session_caching:
			cached = self.__session_cacher.get_cached(state, action, depth)
			if cached is not None:
				return cached

		for destination in states:
			destination_value = self._get_state_value(destination, depth-1)
			transition_probability = self._get_expected_transition_probability(state, action, destination)
			weighted_value = destination_value * transition_probability
			value += weighted_value

		if self.__session_caching:
			self.__session_cacher.cache(
				state, action, value, depth
			)

		return value

	@Logger.logged_method
	def _get_state_value(self, state, depth) -> float:
		reward = self._get_expected_instant_reward(state)

		if depth == 0 or self._is_episode_over(state):
			return reward

		action = self._policy(state, depth)

		value = reward + (self._discount_factor * self._get_state_action_value(state, action, depth))
		return value

	@Logger.logged_method
	def _explore(self, state):
		actions = self._get_available_actions(state)
		# TODO: BETTER EXPLORE FUNCTION. POSSIBLE EXPLORE ACTIONS NEVER TAKEN BEFORE
		return random.choice(actions)

	@Logger.logged_method
	def _exploit(self, state):
		return self._policy(state, self._depth)

	# noinspection PyUnusedLocal
	@Logger.logged_method
	def _get_action(self, state):
		method = None
		if self._explore_exploit_tradeoff == 0:
			method = self._explore
		elif self._explore_exploit_tradeoff == 1:
			method = self._exploit
		else:
			method = random.choice(
				[self._exploit]*int(EXPLORE_EXPLOIT_INSTANCES*self._explore_exploit_tradeoff) +
				[self._explore]*int((1-EXPLORE_EXPLOIT_INSTANCES)*self._explore_exploit_tradeoff)
			)
		return method(state)

	@Logger.logged_method
	def perform_timestep(self):
#		self.__session_cacher.clear()
		state = self._get_environment().get_state()
		action = self._get_action(state)
		self._get_environment().do(action)
		self._update_transition_probability(state, action, self._get_environment().get_state())

	@Logger.logged_method
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
