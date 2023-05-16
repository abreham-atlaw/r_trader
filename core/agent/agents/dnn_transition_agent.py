from typing import *
from abc import ABC

import tensorflow as tf
import numpy as np

import copy

from core import Config
from lib.rl.agent import DNNTransitionAgent
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, AgentState
from core.environment.trade_environment import TradeEnvironment
from core.agent.trader_action import TraderAction
from core.agent.utils.dnn_models import KerasModelHandler


class TraderDNNTransitionAgent(DNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			state_change_delta_model_mode=Config.AGENT_STATE_CHANGE_DELTA_MODEL_MODE,
			state_change_delta=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			update_agent=Config.UPDATE_AGENT,
			depth_mode=Config.AGENT_DEPTH_MODE,
			discount_function=Config.AGENT_DISCOUNT_FUNCTION,
			core_model=None,
			delta_model=None,
			**kwargs
	):
		super().__init__(
			*args,
			depth=Config.AGENT_DEPTH,
			explore_exploit_tradeoff=Config.AGENT_EXPLOIT_EXPLORE_TRADEOFF,
			update_agent=update_agent,
			**kwargs
		)
		self.__state_change_delta_model_mode = state_change_delta_model_mode
		self.__state_change_delta = state_change_delta
		self.__depth_mode = depth_mode
		self.environment: TradeEnvironment

		if core_model is None:
			Logger.info("Loading Core Model")
			core_model = KerasModelHandler.load_model(Config.CORE_MODEL_CONFIG.path)
		self.set_transition_model(core_model)

		self.__delta_model = None
		if state_change_delta_model_mode:
			self.__delta_model = delta_model
			if delta_model is None:
				Logger.info("Loading Delta Model")
				self.__delta_model = KerasModelHandler.load_model(Config.DELTA_MODEL_CONFIG.path)

		self.__state_change_delta_cache = {}
		self.__discount_function = discount_function

	def __check_and_add_depth(self, input_: np.ndarray, depth: int) -> np.ndarray:
		if self.__depth_mode:
			input_ = np.append(input_, depth)
		return input_

	def _state_action_to_model_input(self, state: TradeState, action: TraderAction, final_state: TradeState) -> np.ndarray:
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if not np.all(final_state.get_market_state().get_state_of(base_currency, quote_currency) == state.get_market_state().get_state_of(base_currency, quote_currency)):

				return self.__check_and_add_depth(
					state.get_market_state().get_state_of(base_currency, quote_currency),
					state.get_depth()
				)

		raise ValueError("Initial State and Final state are the same.")  # TODO: FIND ANOTHER WAY TO HANDLE THIS.

	def _get_discount_factor(self, depth) -> float:
		if self.__discount_function is None:
			return super()._get_discount_factor(depth)
		return self.__discount_function(depth)

	def __get_state_change_delta(self, sequence: np.ndarray, direction: int, depth: Optional[int] = None) -> float:

		if direction == -1:
			direction = 0

		model_input = np.append(sequence, direction)
		if depth is not None:
			model_input = self.__check_and_add_depth(model_input, depth)
		model_input = model_input.reshape((1, -1))

		cache_key = model_input.tobytes()

		cached = self.__state_change_delta_cache.get(cache_key)
		if cached is not None:
			return cached

		if self.__state_change_delta_model_mode:
			return_value = self.__delta_model.predict(
				model_input
			).flatten()[0]

		else:
			if isinstance(self.__state_change_delta, float):
				percentage = self.__state_change_delta
			else:
				percentage = np.random.uniform(self.__state_change_delta[0], self.__state_change_delta[1])

			return_value = sequence[-1] * percentage

		self.__state_change_delta_cache[cache_key] = return_value

		return return_value

	def _prediction_to_transition_probability(
			self,
			initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState
	) -> float:
		predicted_value: float = float(tf.reshape(output, (-1,))[0])
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if np.all(final_state.get_market_state().get_state_of(base_currency, quote_currency) == initial_state.get_market_state().get_state_of(base_currency, quote_currency)):
				continue

			if final_state.get_market_state().get_current_price(base_currency, quote_currency) > initial_state.get_market_state().get_current_price(base_currency, quote_currency):
				return predicted_value

			return 1-predicted_value

	def _get_train_output(self, initial_state: TradeState, action, final_state: TradeState) -> np.ndarray:
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if final_state.get_market_state().get_current_price(base_currency, quote_currency) == initial_state.get_market_state().get_current_price(base_currency, quote_currency):
				continue

			if final_state.get_market_state().get_current_price(base_currency, quote_currency) > initial_state.get_market_state().get_current_price(base_currency, quote_currency):
				return np.array([1])

			return np.array([0])

		return np.array([0.5])

	def _get_expected_instant_reward(self, state) -> float:
		return self._get_environment().get_reward(state)

	def __get_involved_instruments(self, open_trades: List[AgentState.OpenTrade]) -> List[Tuple[str, str]]:
		return list(set(
			[
				(open_trade.get_trade().base_currency, open_trade.get_trade().quote_currency)
				for open_trade in open_trades
			]
		))

	def _get_possible_states(self, state: TradeState, action: TraderAction) -> List[TradeState]:

		states = []

		if len(state.get_agent_state().get_open_trades()) != 0:
			states += self.__simulate_instruments_change(
				state,
				self.__get_involved_instruments(state.get_agent_state().get_open_trades())
			)

		elif action is None or action.action == TraderAction.Action.CLOSE:
			states += self.__simulate_instruments_change(
				state,
				state.get_market_state().get_tradable_pairs()
			)

		else:
			states += self.__simulate_instrument_change(state, action.base_currency, action.quote_currency)

		states = [self.__simulate_action(mid_state, action) for mid_state in states]

		return states

	def __simulate_instruments_change(self, mid_state, instruments: List[Tuple[str, str]]) -> List[TradeState]:
		states = []
		for base_currency, quote_currency in instruments:
			states += self.__simulate_instrument_change(mid_state, base_currency, quote_currency)

		return states

	def __simulate_instrument_change(self, state: TradeState, base_currency: str, quote_currency: str) -> List[TradeState]:
		states = []

		original_value = state.get_market_state().get_state_of(base_currency, quote_currency)

		for j in [-1, 1]:
			new_state = state.__deepcopy__()
			new_state.recent_balance = state.get_agent_state().get_balance()
			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				np.array(original_value[-1] + j*self.__get_state_change_delta(original_value, j, state.get_depth())).reshape(1)
			)
			states.append(new_state)

		return states

	def __simulate_action(self, state: TradeState, action: TraderAction) -> TradeState:  # TODO: SETUP CACHER
		new_state = copy.deepcopy(state)

		if action is None:
			return new_state

		if action.action == TraderAction.Action.CLOSE:
			new_state.get_agent_state().close_trades(action.base_currency, action.quote_currency)
			return new_state

		new_state.get_agent_state().open_trade(
			action,
			state.get_market_state().get_current_price(action.base_currency, action.quote_currency)
		)

		return new_state
