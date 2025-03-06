import copy
from typing import *

import numpy as np

import math
from dataclasses import dataclass

from lib.utils.logger import Logger
from lib.rl.environment import ModelBasedState
from core.agent.trader_action import TraderAction
from core import Config


class MarketState:

	def __init__(
			self,
			currencies=None,
			state=None,
			spread_state=None,
			memory_len=None,
			tradable_pairs=None
	):

		self.__currencies = currencies
		if currencies is None:
			self.__currencies = Config.CURRENCIES

		if state is None and memory_len is None:
			raise Exception("Insufficient information given on State.")

		if state is None:
			self.__state = self.__init_state(len(currencies), memory_len)
		else:
			self.__state = state

		self.__tradable_pairs: List[Tuple[str, str]] = tradable_pairs
		if tradable_pairs is None:
			self.__tradable_pairs = [
				(base_currency, quote_currency)
				for base_currency in self.__currencies
				for quote_currency in self.__currencies
				if base_currency != quote_currency
			]

		self.__spread_state = spread_state
		if spread_state is None:
			self.__spread_state = np.zeros((len(currencies), len(currencies))).astype('float64')

	def __init_state(self, num_currencies, memory_len) -> np.ndarray:
		state = np.zeros((num_currencies, num_currencies, memory_len)).astype('float64')
		for i in range(num_currencies):
			state[i, i] = 1
		return state

	def __get_currencies_position(self, base_currency, quote_currency):
		if base_currency not in self.__currencies:
			raise CurrencyNotFoundException(base_currency)
		if quote_currency not in self.__currencies:
			raise CurrencyNotFoundException(quote_currency)

		return self.__currencies.index(base_currency), self.__currencies.index(quote_currency)

	def get_state_of(self, base_currency, quote_currency) -> np.ndarray:
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		return self.__state[bci, qci]

	def get_current_price(self, base_currency, quote_currency) -> np.float32:
		return self.get_state_of(base_currency, quote_currency)[-1]

	def get_spread_state_of(self, base_currency, quote_currency) -> float:
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		return self.__spread_state[bci, qci]

	def update_state_of(self, base_currency, quote_currency, values: np.ndarray):
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		self.__state[bci, qci] = np.concatenate((self.__state[bci, qci, len(values):], values))
		self.__state[qci, bci] = 1/self.__state[bci, qci]

	def update_state_layer(self, state_layer: np.ndarray):
		for i in range(state_layer.shape[0]):
			for j in range(state_layer.shape[1]):
				if not np.isclose(state_layer[i, j], 1/state_layer[j, i]):
					Logger.warning(f"Inconsistent Layer given. {state_layer[i, j], state_layer[j, i]}")

		self.__state[:, :, :-1] = self.__state[:, :, 1:]
		self.__state[:, :, -1] = state_layer

		for i in range(len(self.__currencies)):
			self.__state[i, i, -1] = 1

	def update_spread_state_of(self, base_currency, quote_currency, value: float):
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		self.__spread_state[bci, qci] = value
		self.__spread_state[qci, bci] = self.convert(value, to=base_currency, from_=quote_currency)

	def set_spread_state(self, spread_state: np.ndarray):
		self.__spread_state = spread_state

	def get_price_matrix(self) -> np.ndarray:
		return self.__state

	def get_spread_matrix(self) -> np.ndarray:
		return self.__spread_state

	def get_memory_len(self) -> int:
		return self.__state.shape[2]

	def get_currencies(self) -> List[str]:
		return self.__currencies

	def get_tradable_pairs(self) -> List[Tuple[str, str]]:
		return self.__tradable_pairs

	def convert(self, value, from_, to):
		return value * self.get_current_price(from_, to)

	def __deepcopy__(self, memo=None):
		return MarketState(
			currencies=self.__currencies.copy(),
			tradable_pairs=self.__tradable_pairs.copy(),
			state=self.__state.copy(),
			spread_state=self.__spread_state.copy()
		)

	def __hash__(self):
		return hash((tuple(self.__currencies), tuple(self.__tradable_pairs), self.__state.tobytes(), self.__spread_state.tobytes()))


	def __eq__(self, other):
		if not isinstance(other, MarketState):
			return False
		return \
			(self.__currencies == other._MarketState__currencies) and \
			(self.__tradable_pairs == other._MarketState__tradable_pairs) and \
			(np.all(self.__state == other._MarketState__state)) and \
			(np.all(self.__spread_state == other._MarketState__spread_state))


class AgentState:
	
	class OpenTrade:

		def __init__(self, trade: TraderAction, enter_value, current_value=None):
			self.__trade = trade
			self.__enter_value = enter_value
			self.__current_value = current_value
			if current_value is None:
				self.__current_value = enter_value

		def update_current_value(self, value):
			self.__current_value = value

		def get_unrealized_profit(self) -> float:  # RETURNS PROFIT IN TERMS OF QUOTE CURRENCY
			profit = (self.__current_value - self.__enter_value) * self.__trade.units

			if self.__trade.action == TraderAction.Action.SELL:
				profit *= -1

			return profit

		def get_return(self) -> float:
			return self.__trade.margin_used + self.get_unrealized_profit()

		def get_trade(self) -> TraderAction:
			return self.__trade

		def get_enter_value(self) -> float:
			return self.__enter_value

		def get_current_value(self) -> float:
			return self.__current_value

		def __deepcopy__(self, memo=None):
			return AgentState.OpenTrade(
				trade=self.__trade.__deepcopy__(),
				enter_value=self.__enter_value,
				current_value=self.__current_value
			)

		def __hash__(self):
			return hash((self.__trade, self.__enter_value, self.__current_value))

		def __eq__(self, other):
			return isinstance(other, AgentState.OpenTrade) and \
					self.get_current_value() == other.get_current_value() and \
					self.get_enter_value() == other.get_enter_value() and \
					self.get_trade() == other.get_trade()

	def __init__(
			self,
			balance,
			market_state: MarketState,
			margin_rate=1,
			currency=Config.AGENT_CURRENCY,
			open_trades=None,
			core_pricing=Config.AGENT_CORE_PRICING,
			commission_cost=Config.AGENT_COMMISSION_COST,
			trade_penalty=Config.AGENT_TRADE_PENALTY
	):
		self.__balance = balance
		self.__market_state = market_state
		self.__currency = currency
		self.__margin_rate = margin_rate
		self.__core_pricing = core_pricing
		self.__commission_cost = commission_cost
		self.__trade_penalty = trade_penalty
		self.__open_trades: List[AgentState.OpenTrade] = open_trades
		if open_trades is None:
			self.__open_trades: List[AgentState.OpenTrade] = []

	def __update_open_trades(self):
		for trade in self.__open_trades:
			trade.update_current_value(
				self.__market_state.get_current_price(
					trade.get_trade().base_currency,
					trade.get_trade().quote_currency
				)
			)

	def calc_required_margin(self, units: int, base_currency: str, quote_currency: str) -> float:
		price = self.__market_state.get_current_price(base_currency, quote_currency)
		in_quote = price*self.__margin_rate*units
		return in_quote * self.__market_state.get_current_price(quote_currency, self.__currency)

	def __units_for(self, margin: float, base_currency: str, quote_currency: str) -> int:
		price = self.__market_state.get_current_price(base_currency, quote_currency)
		in_quote = margin * self.__market_state.get_current_price(self.__currency, quote_currency)
		return math.floor(in_quote/(self.__margin_rate * price + Config.DEFAULT_EPSILON))

	def get_balance(self, original=False):
		if original:
			return self.__balance
		self.__update_open_trades()
		return self.__balance + sum([
			self.to_agent_currency(
				value=trade.get_unrealized_profit(),
				from_currency=trade.get_trade().quote_currency
			)
			for trade in self.__open_trades
		])

	def set_balance(self, balance):
		self.__balance = balance

	def update_balance(self, delta):
		self.__balance += delta

	def get_margin_used(self) -> float:
		return sum([trade.get_trade().margin_used for trade in self.__open_trades])

	def get_margin_available(self):
		return self.get_balance(original=True) - self.get_margin_used()

	def get_open_trades(self, base_currency=None, quote_currency=None) -> List[OpenTrade]:

		return [
			trade
			for trade in self.__open_trades
			if (trade.get_trade().base_currency == base_currency or base_currency is None) and
				(trade.get_trade().quote_currency == quote_currency or quote_currency is None)
		]

	def open_trade(self, action: TraderAction, current_value: float = None):
		if current_value is None:
			current_value = self.__market_state.get_current_price(action.base_currency, action.quote_currency)
		if action.margin_used is None:
			action.margin_used = self.calc_required_margin(action.units, action.base_currency, action.quote_currency)
		elif action.units is None:
			action.units = self.__units_for(action.margin_used, action.base_currency, action.quote_currency)
		if action.margin_used > self.get_margin_available():
			raise InsufficientFundsException

		enter_value = current_value + ((action.action - 0.5) * 2) * self.__market_state.get_spread_state_of(action.base_currency, action.quote_currency)

		# self.__balance -= action.units * self.__to_agent_currency(
		# 	value=self.__market_state.get_spread_state_of(action.base_currency, action.quote_currency),
		# 	from_currency=action.quote_currency
		# )

		if self.__core_pricing:
			self.__balance -= self.__commission_cost

		self.__balance -= self.__trade_penalty

		self.__open_trades.append(
			AgentState.OpenTrade(action, enter_value)
		)

	def to_agent_currency(self, value, from_currency) -> float:
		return self.__market_state.convert(value, to=self.__currency, from_=from_currency)

	def add_open_trade(self, trade: OpenTrade):
		self.__open_trades.append(trade)

	def close_trades(self, base_currency, quote_currency, modify_balance=True):
		self.__update_open_trades()
		open_trades = self.get_open_trades(base_currency, quote_currency)
		if modify_balance:
			self.update_balance(
				sum([
					self.to_agent_currency(
						value=trade.get_unrealized_profit(),
						from_currency=trade.get_trade().quote_currency
					)
					for trade in open_trades
				])
			)
		self.__open_trades = [trade for trade in self.__open_trades if trade not in open_trades]

	def __deepcopy__(self, memo=None):
		if memo is None:
			memo = {}

		market_state = memo.get('market_state')
		if market_state is None:
			market_state = copy.deepcopy(self.__market_state)

		open_trades = [trade.__deepcopy__() for trade in self.__open_trades]

		return AgentState(
			self.__balance,
			market_state,
			margin_rate=self.__margin_rate,
			currency=self.__currency,
			open_trades=open_trades
		)

	def __hash__(self):
		return hash((self.__balance, tuple(self.__open_trades)))


class TradeState(ModelBasedState):

	def __init__(self, market_state: MarketState = None, agent_state: AgentState = None, recent_balance: float = None):
		self.market_state = market_state
		self.agent_state = agent_state
		self.recent_balance = recent_balance
		self.__depth = 0
		self.__attached_state = {}

	def get_market_state(self) -> MarketState:
		return self.market_state

	def get_agent_state(self) -> AgentState:
		return self.agent_state

	def get_recent_balance(self) -> float:
		return self.recent_balance

	def get_recent_balance_change(self) -> float:
		if self.get_recent_balance() is None:
			return 0
		return self.get_agent_state().get_balance() - self.get_recent_balance()

	def set_depth(self, depth: int):
		self.__depth = depth

	def get_depth(self) -> int:
		return self.__depth

	def attach_state(self, key: Hashable, state: Any):
		self.__attached_state[key] = state

	def detach_state(self, key: Hashable) -> Any:
		return self.__attached_state.pop(key)

	def get_attached_state(self, key: Hashable) -> Any:
		return self.__attached_state[key]

	def is_state_attached(self, key: Hashable) -> bool:
		return key in self.__attached_state.keys()

	def __deepcopy__(self, memo=None):
		market_state = self.market_state.__deepcopy__()
		agent_state = self.agent_state.__deepcopy__(memo={'market_state': market_state})

		return TradeState(market_state, agent_state, self.get_recent_balance())

	def __hash__(self):
		return hash((self.market_state, self.agent_state, self.get_recent_balance()))

	def __eq__(self, other):
		if not isinstance(other, TraderAction):
			return False


@dataclass
class ArbitradeTradeState:

	class MarginStage:
		STAGE_ZERO = 0
		STAGE_ONE = 1

	start_point: float
	checkpoints: Tuple[float, float]
	close_points: Tuple[float, float]
	instrument: Tuple[str, str]
	margin_stage: int = MarginStage.STAGE_ZERO

	def increment_margin_stage(self):
		if self.margin_stage == ArbitradeTradeState.MarginStage.STAGE_ONE:
			return
		self.margin_stage += 1


class CurrencyNotFoundException(Exception):

	def __init__(self, currency):
		self.currency = currency

	def __str__(self):
		return "Currency not found: " + self.currency


class InsufficientFundsException(Exception):
	pass
