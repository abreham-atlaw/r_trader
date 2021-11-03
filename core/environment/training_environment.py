from typing import *

import numpy as np

import psycopg2 as pg

import datetime

from .trade_environment import TradeEnvironment
from core.environment.trade_state import TradeState, AgentState, MarketState
from core import Config


class TrainingEnvironment(TradeEnvironment):

	def __init__(self, data_path, initial_balance=100, pg_config=None, table_name=Config.TABLE_NAME,
					datetime_format="%Y-%m-%d %H:%M:*"):
		super(TrainingEnvironment, self).__init__()
		self.__data_path = data_path
		self.__initial_balance = initial_balance
		if pg_config is None:
			self.__pg_config = Config.DEFAULT_PG_CONFIG
		else:
			self.__pg_config = pg_config
		self.__table_name = table_name
		self.__pg_connection = pg.connect(**self.__pg_config)
		self.datetime_format = datetime_format
		self.last_datetime = None

	def _initiate_state(self) -> TradeState:
		agent_state = AgentState(self.__initial_balance)
		market_state = self.__initiate_market_state(Config.MARKET_STATE_MEMORY)
		return TradeState(agent_state=agent_state, market_state=market_state)

	def __prepare_timestep_state_layer(self, currencies: List[str], datetime_: datetime.datetime) -> np.ndarray:
		state_layer = np.zeros((len(currencies), len(currencies)))

		cursor = self.__pg_connection.cursor()

		cursor.execute(
			f"SELECT base_currency, quote_currency, close FROM {self.__table_name} WHERE datetime='{datetime_.strftime(self.datetime_format)};"
		)

		for base_currency, quote_currency, value in cursor.fetchall():

			if base_currency not in currencies or quote_currency not in currencies:
				continue

			state_layer[
				currencies.index(base_currency),
				currencies.index(quote_currency)
			] = value

		cursor.close()

		return state_layer

	def __initiate_market_state(self, memory_size) -> MarketState:

		currencies = []
		tradable_pairs = []

		cursor = self.__pg_connection.cursor()
		cursor.execute(f"SELECT DISTINCT base_currency, quote_currency FROM {self.__table_name};")
		for base_currency, quote_currency in cursor.fetchall():
			if base_currency not in currencies:
				currencies.append(base_currency)
			if quote_currency not in currencies:
				currencies.append(quote_currency)

			if (base_currency, quote_currency) not in tradable_pairs:
				tradable_pairs.append((base_currency, quote_currency))

		market_state = MarketState(currencies=currencies, memory_len=memory_size, tradable_pairs=tradable_pairs)

		datetimes = cursor.execute(
			f"SELECT DISTINCT datetime FROM {self.__table_name} ORDER BY datetime ASC LIMIT {memory_size};")

		for i, (datetime_,) in enumerate(datetimes):
			market_state.update_state_layer(
				self.__prepare_timestep_state_layer(currencies, datetime_)
			)

		self.last_datetime = datetimes[-1][0]

		cursor.close()

		return market_state

	def _refresh_state(self):
		cursor = self.__pg_connection.cursor()
		cursor.execute(
			f"SELECT DISTINCT datetime FROM {self.__table_name} WHERE datetime > {self.last_datetime.strftime(self.datetime_format)} ORDER BY datetime ASC LIMIT 1;")
		(next_datetime_,) = cursor.fetchone()
		self._state.market_state.update_state_layer(
			self.__prepare_timestep_state_layer(
				self._state.market_state.get_currencies(),
				next_datetime_
			)
		)
