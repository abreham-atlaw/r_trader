from typing import *
from abc import ABC, abstractmethod

import os

from .mba import ModelBasedAgent


class MarkovAgent(ModelBasedAgent, ABC):

	def __init__(self, *args, **kwargs):
		super(MarkovAgent, self).__init__(*args, **kwargs)

	def _get_state_value(self, state, depth) -> float:
		reward = self._get_expected_instant_reward(state)

		if depth == 0 or self._is_episode_over(state):
			return reward

		action = self._policy(state, depth=depth)
		value = reward + (self._get_discount_factor(depth) * self._get_state_action_value(state, action, depth=depth))
		return value

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		states = self._get_possible_states(state, action)
		value = 0

		depth = kwargs.get("depth")
		if depth is None:
			depth = self._depth

		for destination in states:
			destination_value = self._get_state_value(destination, depth-1)
			transition_probability = self._get_expected_transition_probability(state, action, destination)
			weighted_value = destination_value * transition_probability
			value += weighted_value

		return value
