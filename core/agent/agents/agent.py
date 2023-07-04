

from .action_choice_trader import ActionChoiceBalancerTrader
from .action_choice_trader.action_choice_trader import ActionChoiceTrader
from .dnn_transition_agent import TraderDNNTransitionAgent
from .montecarlo_agent import TraderMonteCarloAgent


class TraderAgent(TraderDNNTransitionAgent, TraderMonteCarloAgent, ActionChoiceTrader):
	pass
