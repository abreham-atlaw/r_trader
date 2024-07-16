from abc import ABC

import numpy as np

from datetime import datetime

from lib.rl.agent import MonteCarloAgent
from lib.rl.agent.dra.dra import DeepReinforcementAgent
from temp import stats

from lib.rl.environment import ModelBasedState


class DeepReinforcementMonteCarloAgent(MonteCarloAgent, DeepReinforcementAgent, ABC):

	def __init__(self, *args, wp: float = 1, **kwargs):
		super().__init__(*args, **kwargs)
		self.__wp = wp

	@staticmethod
	def weighted_sigmoid(x, w) -> float:
		return 1 / (1 + np.exp(-np.dot(x, w)))

	def __calc_value(self, calculated, predicted, visits) -> float:
		return predicted + self.weighted_sigmoid(visits, self.__wp)*(calculated - predicted)

	def _get_action_node_value(self, node: 'MonteCarloAgent.Node'):
		if node.predicted_value is None:
			start_time = datetime.now()
			node.predicted_value = DeepReinforcementAgent._get_state_action_value(
				self,
				self._state_repository.retrieve(node.parent.id),
				node.action
			)
			stats.durations["DeepReinforcementAgent._get_state_action_value"] += (datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		calculated_value = super()._get_action_node_value(node)
		stats.durations["super()._get_action_node_value"] += (datetime.now() - start_time).total_seconds()

		value = self.__calc_value(calculated_value, node.predicted_value, node.get_visits())
		return value

	def _update_state_action_value(self, initial_state: ModelBasedState, action, final_state: ModelBasedState, value):
		DeepReinforcementAgent._update_state_action_value(self, initial_state, action, final_state, value)
