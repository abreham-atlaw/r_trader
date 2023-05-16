

from .action_choice_trader import ActionChoiceBalancerTrader
from .dnn_transition_agent import TraderDNNTransitionAgent
from .montecarlo_agent import TraderMonteCarloAgent


class TraderAgent(ActionChoiceBalancerTrader, TraderDNNTransitionAgent, TraderMonteCarloAgent):
	pass
