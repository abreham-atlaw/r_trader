from typing import *

import numpy as np
import pandas as pd

from datetime import datetime

from lib.utils.logger import Logger
from lib.network.oanda import Trader
from lib.network.oanda.data import models
from core import Config
from core.environment.trade_state import TradeState, AgentState, MarketState
from core.agent.trader_action import TraderAction
from core.environment.trade_environment import TradeEnvironment


class LiveEnvironment(TradeEnvironment):

	def __init__(self):
		super(LiveEnvironment, self).__init__()
		self.__trader = Trader(
			Config.OANDA_TOKEN,
			Config.OANDA_TRADING_ACCOUNT_ID
		)

	def __to_oanda_action(self, action):
		if action == TraderAction.Action.BUY:
			return Trader.TraderAction.BUY
		if action == TraderAction.Action.SELL:
			return Trader.TraderAction.SELL

	def __from_oanda_action(self, action):
		if action == Trader.TraderAction.BUY:
			return TraderAction.Action.BUY
		if action == Trader.TraderAction.SELL:
			return TraderAction.Action.SELL

	def __get_agent_state(self, market_state: MarketState) -> AgentState:
		balance = self.__trader.get_balance()
		open_trades_raw: List[models.Trade] = self.__trader.get_open_trades()
		open_trades: List[AgentState.OpenTrade] = []
		for trade in open_trades_raw:
			base_currency, quote_currency = trade.get_instrument()
			open_trades.append(
				AgentState.OpenTrade(
					TraderAction(
						base_currency,
						quote_currency,
						self.__from_oanda_action(trade.get_action()),
						margin_used=trade.marginUsed,
						units=trade.get_units()
					),
					trade.get_initial_price(),
					current_value=trade.price
				)
			)
		return AgentState(balance, market_state, open_trades=open_trades, margin_rate=self.__trader.get_margin_rate())

	def __get_currencies(self, pairs: List[Tuple[str, str]]) -> List[str]:
		currencies = []
		for pair in pairs:
			currencies += pair
		return list(set(currencies))

	def __get_market_state(self, memory_size) -> MarketState:
		tradeable_pairs = self.__trader.get_instruments()
#		tradeable_pairs = [tradeable_pairs[i] for i in np.random.choice(len(tradeable_pairs), 5, False)]
#		print(f"Tradeable Pairs: {tradeable_pairs}")
		tradeable_pairs = [
			("AUD", "USD"),
			("EUR", "GBP"),
			("EUR", "AUD"),
			("GBP", "AUD"),
			("USD", "JPY")
		]

		market_state = MarketState(
			currencies=self.__get_currencies(tradeable_pairs),
			tradable_pairs=tradeable_pairs,
			memory_len=memory_size
		)

		for base_currency, quote_currency in tradeable_pairs:
			market_state.update_state_of(
				base_currency,
				quote_currency,
				self.__prepare_tradable_pairs(base_currency, quote_currency, memory_size)
			)

		return market_state

	def __candlesticks_to_dataframe(self, candlesticks: List[models.CandleStick]) -> pd.DataFrame:
		df_list = []
		for candlestick in candlesticks:
			candle_dict = candlestick.mid
			candle_dict["v"] = candlestick.volume
			df_list.append(candle_dict)

		return pd.DataFrame(df_list)

	def __prepare_tradable_pairs(self, base_currency, quote_currency, size) -> np.ndarray:
		candle_sticks = self.__trader.get_candlestick(
			(base_currency, quote_currency),
			count=size,
			to=datetime.now(),
			granularity="M1"
		)
		df = self.__candlesticks_to_dataframe(candle_sticks)
		return df["c"].to_numpy()

	@Logger.logged_method
	def _open_trade(self, action: TraderAction):
		super()._open_trade(action)
		self.__trader.trade(
			(action.base_currency, action.quote_currency),
			self.__to_oanda_action(action.action),
			action.margin_used
		)

	@Logger.logged_method
	def _close_trades(self, base_currency, quote_currency):
		super()._close_trades(base_currency, quote_currency)
		self.__trader.close_trades((base_currency, quote_currency))

	@Logger.logged_method
	def _initiate_state(self) -> TradeState:
		market_state = self.__get_market_state(Config.MARKET_STATE_MEMORY)
		agent_state = self.__get_agent_state(market_state)

		return TradeState(
			agent_state=agent_state,
			market_state=market_state
		)

	@Logger.logged_method
	def _refresh_state(self, state: TradeState = None) -> TradeState:
		if state is None:
			state = self.get_state()
		return self._initiate_state()
