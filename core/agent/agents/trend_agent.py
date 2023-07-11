import typing

import numpy as np

from core import Config
from core.agent.trader_action import TraderAction
from core.environment.trade_state import TradeState
from lib.rl.agent import Agent


class TrendTraderAgent(Agent):

	def __init__(self, *args, trade_size=Config.AGENT_RANDOM_TRADE_SIZE, **kwargs):
		super().__init__(*args, **kwargs)
		self.__trade_size = trade_size

	def __get_action(self, data: np.array) -> typing.Optional[int]:
		ma50 = np.mean(data[-50:])
		ma200 = np.mean(data[-200:])

		if ma50 > ma200:
			signal = TraderAction.Action.BUY
		elif ma50 < ma200:
			signal = TraderAction.Action.SELL
		else:
			signal = None

		return signal

	def _policy(self, state: TradeState) -> typing.Optional[TraderAction]:
		if len(state.get_agent_state().get_open_trades()) > 0:
			return None

		for base_currency, quote_currency in state.get_market_state().get_tradable_pairs():
			action = self.__get_action(state.get_market_state().get_state_of(base_currency, quote_currency))
			if action is None:
				pass
			return TraderAction(
				base_currency=base_currency,
				quote_currency=quote_currency,
				action=action,
				margin_used=state.get_agent_state().get_margin_available() * self.__trade_size
			)
		return None
