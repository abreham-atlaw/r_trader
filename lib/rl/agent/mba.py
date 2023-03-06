from typing import *
from abc import ABC, abstractmethod

import numpy as np

from lib.rl.environment import ModelBasedState
from .agent import Agent


class ModelBasedAgent(Agent, ABC):

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

	def __init__(self, discount: float = 0.7, depth: int = None, session_caching: bool = True, **kwargs):
		super(ModelBasedAgent, self).__init__(**kwargs)
		self._discount_factor = discount
		self._depth = depth
		self.__session_caching = session_caching
		if session_caching:
			self.__session_cacher: ModelBasedAgent.Cacher = ModelBasedAgent.Cacher()

	@abstractmethod
	def _get_expected_transition_probability(self, initial_state: ModelBasedState, action, final_state) -> float:
		pass

	@abstractmethod
	def _update_transition_probability(self, initial_state: ModelBasedState, action, final_state):
		pass

	@abstractmethod
	def _get_expected_instant_reward(self, state) -> float:
		pass

	@abstractmethod
	def _get_possible_states(self, state: ModelBasedState, action) -> List[ModelBasedState]:
		pass

	def _update_state_action_value(self, initial_state: ModelBasedState, action, final_state: ModelBasedState, value):
		self._update_transition_probability(initial_state, action, final_state)

	