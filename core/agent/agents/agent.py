from lib.rl.agent.cra import CumulativeRewardAgent
from .action_choice_trader.action_choice_trader import ActionChoiceTrader
from .cra import CumulativeRewardTraderAgent
from .drmca import TraderDeepReinforcementMonteCarloAgent
from .montecarlo_agent import TraderMonteCarloAgent


class TraderAgent(
	CumulativeRewardTraderAgent,
	TraderDeepReinforcementMonteCarloAgent,
	TraderMonteCarloAgent,
	ActionChoiceTrader
):
	pass
