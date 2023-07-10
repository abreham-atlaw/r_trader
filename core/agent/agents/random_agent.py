import typing

from core import Config
from core.agent.trader_action import TraderAction
from core.environment.trade_state import TradeState
from lib.rl.agent import Agent
import random


class TraderRandomAgent(Agent):

	def __init__(self, *args, trade_size=Config.AGENT_RANDOM_TRADE_SIZE, **kwargs):
		super().__init__(*args, **kwargs)
		self.__trade_size = trade_size

	def _policy(self, state: TradeState) -> typing.Optional[TraderAction]:
		if len(state.get_agent_state().get_open_trades()) > 0:
			return None

		instrument = random.choice(state.get_market_state().get_tradable_pairs())
		direction = random.choice([TraderAction.Action.BUY, TraderAction.Action.SELL])
		size = state.get_agent_state().get_margin_available() * self.__trade_size
		return TraderAction(
			instrument[0],
			instrument[1],
			direction,
			margin_used=size
		)
