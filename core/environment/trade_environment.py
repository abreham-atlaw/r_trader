from typing import *
from abc import abstractmethod, ABC

import numpy as np

from lib.utils.logger import Logger
from .trade_state import TradeState
from core.agent.trader_action import TraderAction
from lib.rl.environment import Environment


class TradeEnvironment(Environment, ABC):

	def __init__(self, time_penalty=-1, trade_size_gap=10, market_state_memory=64):
		super(TradeEnvironment, self).__init__()
		self._state: TradeState = None
		self.__time_penalty = time_penalty
		self.__trade_size_gap = trade_size_gap
		self.market_state_memory = market_state_memory

	@abstractmethod
	def _initiate_state(self) -> TradeState:
		pass

	@abstractmethod
	def _refresh_state(self, state: TradeState = None) -> TradeState:
		pass

	def _initialize(self):
		super()._initialize()
		self._state = self._initiate_state()

	@Logger.logged_method
	def _close_trades(self, base_currency, quote_currency):
		self.get_state().get_agent_state().close_trades(base_currency, quote_currency)

	@Logger.logged_method
	def _open_trade(self, action: TraderAction):
		self.get_state().get_agent_state().open_trade(
			action
		)

	@Logger.logged_method
	def get_reward(self, state: TradeState or None = None):
		if state is None:
			state = self.get_state()
		return state.get_agent_state().get_balance() + self.__time_penalty

	@Logger.logged_method
	def perform_action(self, action: TraderAction):

		if action.action == TraderAction.Action.CLOSE:
			self._close_trades(action.base_currency, action.quote_currency)
			return
		self._open_trade(action)

		self._state = self._refresh_state()

	def render(self):
		pass

	def update_ui(self):
		pass

	def check_is_running(self) -> bool:
		return True

	@Logger.logged_method
	def get_valid_actions(self, state=None) -> List[Union[TraderAction, None]]:
		if state is None:
			state = self.get_state()
		pairs = state.get_market_state().get_tradable_pairs()
		pairs = [pairs[i] for i in np.random.choice(len(pairs), 5, False)]
		amounts = [
			(i + 1) * self.__trade_size_gap
			for i in range(int(state.get_agent_state().get_margin_available() // self.__trade_size_gap))
		]

		actions = [
			TraderAction(
				pair[0],
				pair[1],
				action,
				margin_used=amount
			)
			for pair in pairs
			for action in [TraderAction.Action.BUY, TraderAction.Action.SELL]
			for amount in amounts
		]

		actions += [
			TraderAction(trade.get_trade().base_currency, trade.get_trade().quote_currency, TraderAction.Action.CLOSE)
			for trade in state.get_agent_state().get_open_trades()
		]

		# actions.append(None)

		return actions

	def get_state(self) -> TradeState:
		if self._state is None:
			raise Exception("State not Initialized.")
		return self._state

	def is_episode_over(self, state=None) -> bool:
		return False
