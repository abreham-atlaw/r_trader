from abc import ABC
from datetime import datetime

from lib.rl.agent import MonteCarloAgent
from lib.rl.agent.dra.dra import DeepReinforcementAgent
from temp import stats

from lib.rl.environment import ModelBasedState


class DeepReinforcementMonteCarloAgent(MonteCarloAgent, DeepReinforcementAgent, ABC):

	def __init__(self, *args, wp: float = 50, **kwargs):
		super().__init__(*args, **kwargs)
		self.__wp = wp

	def _get_action_node_value(self, node: 'MonteCarloAgent.Node'):
		start_time = datetime.now()
		predicted_value = DeepReinforcementAgent._get_state_action_value(
			self,
			self._state_repository.retrieve(node.parent.id),
			node.action
		)
		stats.durations["DeepReinforcementAgent._get_state_action_value"] += (datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		calculated_value = super()._get_action_node_value(node)
		stats.durations["super()._get_action_node_value"] += (datetime.now() - start_time).total_seconds()

		value = ((self.__wp - node.visits)*predicted_value + node.visits*calculated_value)/self.__wp
		return value

	def _update_state_action_value(self, initial_state: ModelBasedState, action, final_state: ModelBasedState, value):
		DeepReinforcementAgent._update_state_action_value(self, initial_state, action, final_state, value)
