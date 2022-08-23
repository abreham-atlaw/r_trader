from typing import *
from abc import ABC

import tensorflow as tf
import numpy as np

from datetime import datetime
import copy
from dataclasses import dataclass

from core import Config
from lib.rl.agent import DNNTransitionAgent, MarkovAgent, MonteCarloAgent
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, AgentState
from core.environment.trade_environment import TradeEnvironment
from .trader_action import TraderAction
from .trade_transition_model import TransitionModel
from .stm import TraderNodeShortTermMemory


class TraderDNNTransitionAgent(DNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
			state_change_delta_model_mode=Config.AGENT_STATE_CHANGE_DELTA_MODEL_MODE,
			state_change_delta=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			update_agent=Config.UPDATE_AGENT,
			**kwargs
	):
		super().__init__(
			*args,
			episodic=False,
			depth=Config.AGENT_DEPTH,
			explore_exploit_tradeoff=Config.AGENT_EXPLOIT_EXPLORE_TRADEOFF,
			update_agent=update_agent,
			**kwargs
		)
		self.__trade_size_gap = trade_size_gap
		self.__state_change_delta_model_mode = state_change_delta_model_mode
		self.__state_change_delta = state_change_delta
		self.environment: TradeEnvironment
		Logger.info("Loading Core Model")
		self.set_transition_model(
			TransitionModel.load_model(Config.CORE_MODEL_CONFIG.path)
		)
		self.__state_change_delta_cache = {}
		self.__delta_model = None
		if state_change_delta_model_mode:
			Logger.info("Loading Delta Model")
			self.__delta_model = TransitionModel.load_model(Config.DELTA_MODEL_CONFIG.path)

	def _state_action_to_model_input(self, state: TradeState, action: TraderAction, final_state: TradeState) -> np.ndarray:
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if not np.all(final_state.get_market_state().get_state_of(base_currency, quote_currency) == state.get_market_state().get_state_of(base_currency, quote_currency)):

				return state.get_market_state().get_state_of(base_currency, quote_currency)

		raise ValueError("Initial State and Final state are the same.") # TODO: FIND ANOTHER WAY TO HANDLE THIS.

	def __get_state_change_delta(self, sequence: np.ndarray, direction) -> float:
		if direction == -1:
			direction = 0

		model_input = np.append(sequence, direction).reshape((1, -1))
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
		mid_state = self.__simulate_action(state, action)

		states = []

		if len(state.get_agent_state().get_open_trades()) != 0:
			states += self.__simulate_instruments_change(
				mid_state,
				self.__get_involved_instruments(state.get_agent_state().get_open_trades())
			)

		elif action is None or action.action == TraderAction.Action.CLOSE:
			states += self.__simulate_instruments_change(
				mid_state,
				state.get_market_state().get_tradable_pairs()
			)

		else:
			states += self.__simulate_instrument_change(mid_state, action.base_currency, action.quote_currency)

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
			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				np.array(original_value[-1] + j*self.__get_state_change_delta(original_value, j)).reshape(1)
			)
			states.append(new_state)

		return states

	def __simulate_action(self, state: TradeState, action: TraderAction) -> TradeState:  # TODO: SETUP CACHER
		new_state = copy.deepcopy(state)
		new_state.recent_balance = state.get_agent_state().get_balance()

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


class TraderMarkovAgent(MarkovAgent, TraderDNNTransitionAgent):
	
	def __init__(self, *args, **kwargs):
		super(TraderMarkovAgent, self).__init__(*args, **kwargs)


class TraderMonteCarloAgent(MonteCarloAgent, TraderDNNTransitionAgent):

	def __init__(
			self,
			*args,
			step_time=Config.AGENT_STEP_TIME,
			discount=Config.AGENT_DISCOUNT_FACTOR,
			min_free_memory_percent=Config.MIN_FREE_MEMORY,
			logical=Config.AGENT_LOGICAL_MCA,
			uct_exploration_weight=Config.AGENT_UCT_EXPLORE_WEIGHT,
			use_stm=Config.AGENT_STM,
			stm_size=Config.AGENT_STM_SIZE,
			stm_threshold=Config.AGENT_STM_THRESHOLD,
			stm_balance_tolerance=Config.AGENT_STM_BALANCE_TOLERANCE,
			stm_average_window=Config.AGENT_STM_AVERAGE_WINDOW_SIZE,
			stm_attention_mode=Config.AGENT_STM_ATTENTION_MODE,
			**kwargs
	):
		super(TraderMonteCarloAgent, self).__init__(
			*args,
			discount=discount,
			min_free_memory_percent=min_free_memory_percent,
			logical=logical,
			uct_exploration_weight=uct_exploration_weight,
			use_stm=use_stm,
			short_term_memory=TraderNodeShortTermMemory(
				stm_size,
				stm_threshold,
				stm_average_window,
				balance_tolerance=stm_balance_tolerance,
				attention_mode=stm_attention_mode
			),
			**kwargs
		)
		self.__step_time = step_time

	def _init_resources(self) -> object:
		start_time = datetime.now()
		return start_time

	def _has_resource(self, start_time) -> bool:
		return (datetime.now() - start_time).seconds < self.__step_time

	def _get_state_node_instant_value(self, state_node: 'MonteCarloAgent.Node') -> float:
		return self._get_environment().get_reward(state_node.state) - self._get_environment().get_reward(state_node.parent.parent.state)
