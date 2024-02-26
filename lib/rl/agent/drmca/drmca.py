from abc import ABC

from lib.rl.agent import MonteCarloAgent
from lib.rl.agent.dra.dra import DeepReinforcementAgent


class DeepReinforcementMonteCarloAgent(MonteCarloAgent, DeepReinforcementAgent, ABC):

	def __init__(self, *args, wp: float = 50, **kwargs):
		super().__init__(*args, **kwargs)
		self.__wp = wp

	def _get_action_node_value(self, node: 'MonteCarloAgent.Node'):
		predicted_value = super(DeepReinforcementAgent, self)._get_state_action_value(
			self._state_repository.retrieve(node.parent.id),
			node.action
		)
		calculated_value = super()._get_action_node_value(node)
		value = ((self.__wp - node.visits)*predicted_value + node.visits*calculated_value)/self.__wp
		return value
