from abc import ABC

from lib.rl.agent import MonteCarloAgent
from lib.rl.agent.dra.dra import DeepReinforcementAgent


class DeepReinforcementMonteCarloAgent(MonteCarloAgent, DeepReinforcementAgent, ABC):

	def _get_action_node_value(self, node: 'MonteCarloAgent.Node'):
		calculated_value = super()._get_action_node_value(node)
		predicted_value = super(DeepReinforcementAgent, self)._get_state_action_value(
			self._state_repository.retrieve(node.parent.id),
			node.action
		)
		self._getroo
		return (node.visits*)


