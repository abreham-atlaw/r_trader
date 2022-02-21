from typing import *

import unittest
from unittest import mock

import numpy as np

from core.environment.trade_state import MarketState, AgentState, TradeState
from core.agent.trader_action import TraderAction


class MarketStateTest(unittest.TestCase):

	CURRENCIES = ["USD", "EUR", "AUD", "CAD", "GBP"]
	MEMORY_LEN = 8

	def setUp(self):
		self.market_state = MarketState(
			memory_len=MarketStateTest.MEMORY_LEN,
			currencies=MarketStateTest.CURRENCIES
		)

	def test_init(self):
		self.assertEqual(
			self.market_state._MarketState__state.shape,
			(len(MarketStateTest.CURRENCIES), len(MarketStateTest.CURRENCIES), MarketStateTest.MEMORY_LEN)
		)

	def test_update_and_get_state(self):
		base_currency = MarketStateTest.CURRENCIES[0]
		quote_currency = MarketStateTest.CURRENCIES[1]

		values0 = np.arange(1, 9)

		self.market_state.update_state_of(base_currency, quote_currency, values0)
		self.assertNotIn(False, self.market_state.get_state_of(base_currency, quote_currency) == values0)

		values1 = np.arange(3, 6)
		self.market_state.update_state_of(base_currency, quote_currency, values1)
		self.assertNotIn(
			False,
			self.market_state.get_state_of(base_currency, quote_currency) == np.concatenate((values1, values0[:MarketStateTest.MEMORY_LEN - len(values1)]))
		)

	def test_update_state_layer(self):
		values = np.zeros((len(MarketStateTest.CURRENCIES), len(MarketStateTest.CURRENCIES)))
		hashes = [abs(hash(currency)) for currency in MarketStateTest.CURRENCIES]

		for i, base_values in enumerate(hashes):
			for j, quote_value in enumerate(hashes):
				values[i, j] = base_values/quote_value

		self.market_state.update_state_layer(values)
		self.assertEqual(
			self.market_state.get_state_of(MarketStateTest.CURRENCIES[0], MarketStateTest.CURRENCIES[1])[0],
			values[0, 1]
		)


class OpenTradeTest(unittest.TestCase):

	ENTER_PRICE = 14.75566
	EXIT_PRICE = 14.76500
	BASE_CURRENCY = "USD"
	QUOTE_CURRENCY = "ZAR"
	ACTION = TraderAction.Action.SELL
	UNREALIZED_PROFIT = -0.95  # IN USD
	CONVERSION_FACTOR = 0.06753
	MARGIN_USED = 105
	UNITS = 1500

	def setUp(self):
		self.trade = AgentState.OpenTrade(
			TraderAction(
				OpenTradeTest.BASE_CURRENCY,
				OpenTradeTest.QUOTE_CURRENCY,
				OpenTradeTest.ACTION,
				margin_used=OpenTradeTest.MARGIN_USED,
				units=OpenTradeTest.UNITS
			),
			OpenTradeTest.ENTER_PRICE
		)

	def test_get_unrealized_profit(self):
		self.trade.update_current_value(OpenTradeTest.EXIT_PRICE)
		up = self.trade.get_unrealized_profit(conversion_factor=OpenTradeTest.CONVERSION_FACTOR)
		self.assertAlmostEqual(
			up,
			OpenTradeTest.UNREALIZED_PROFIT,
			places=1
		)


class AgentStateTest(unittest.TestCase):

	def test_get_margin_available(self):

		market_state = mock.Mock()
		agent_state = AgentState(100, market_state)

		market_state.get_state_of.get_return = [1, 2, 3, 4]
		agent_state.open_trade(
			TraderAction("USD", "CAD", TraderAction.Action.SELL, margin_used=30, units=1000)
		)
		self.assertEqual(
			agent_state.get_margin_available(),
			70
		)

	def test_get_balance(self):

		market_state = mock.MagicMock()
		agent_state = AgentState(96.7, market_state)

		market_state.get_state_of = mock.MagicMock(side_effect=[[7.717], [7.728], [0.00881]])
		agent_state.open_trade(
			TraderAction("ZAR", "JPY", TraderAction.Action.SELL, margin_used=67.92, units=10000)
		)

		self.assertAlmostEqual(
			agent_state.get_balance(),
			95.73,
			places=2
		)

	def test_close_trades(self):
		market_state = mock.MagicMock()
		agent_state = AgentState(96.7, market_state)
		market_state.get_state_of = mock.MagicMock(side_effect=[[7.717], [7.728], [0.00881]])
		agent_state.open_trade(
			TraderAction("ZAR", "JPY", TraderAction.Action.SELL, margin_used=67.92, units=10000)
		)

		agent_state.close_trades("ZAR", "JPY")
		self.assertAlmostEqual(
			agent_state.get_balance(original=True),
			95.73,
			places=2
		)


if __name__ == "__main__":
	unittest.main()
