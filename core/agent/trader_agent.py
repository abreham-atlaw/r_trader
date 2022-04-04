from typing import *
from abc import ABC

import tensorflow as tf
import numpy as np

from datetime import datetime
import copy

from core import Config
from lib.rl.agent import DNNTransitionAgent, MarkovAgent, MonteCarloAgent
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState
from core.environment.trade_environment import TradeEnvironment
from .trader_action import TraderAction
from .trade_transition_model import TransitionModel, RemoteTransitionModel


class TraderDNNTransitionAgent(DNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
			state_change_delta=Config.AGENT_STATE_CHANGE_DELTA,
			**kwargs
	):
		super().__init__(
			*args,
			episodic=False,
			depth=Config.AGENT_DEPTH,
			explore_exploit_tradeoff=Config.AGENT_EXPLOIT_EXPLORE_TRADEOFF,
			**kwargs
		)
		self.__trade_size_gap = trade_size_gap
		self.__state_change_delta = state_change_delta
		self.environment: TradeEnvironment
		if Config.REMOTE_MODEL:
			Logger.info("Using Remote Model")
			self.set_transition_model(
				RemoteTransitionModel(Config.REMOTE_TRANSITION_MODEL_ADDRESS)
			)
		elif Config.NEW_MODEL:
			Logger.info("Creating new Model")
			self.set_transition_model(
				TransitionModel()
			)
		else:
			Logger.info("Loading Model")
			self.set_transition_model(
				TransitionModel.load_model(Config.MODEL_PATH)
			)

	def _state_action_to_model_input(self, state: TradeState, action: TraderAction, final_state: TradeState) -> np.ndarray:
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if final_state.get_market_state().get_state_of(base_currency, quote_currency)[0] != state.get_market_state().get_state_of(base_currency, quote_currency)[0]:

				return state.get_market_state().get_state_of(base_currency, quote_currency)

		raise ValueError("Initial State and Final state are the same.")

	def __get_state_change_delta(self) -> float:
		if isinstance(self.__state_change_delta, float):
			return self.__state_change_delta
		return np.random.uniform(self.__state_change_delta[0], self.__state_change_delta[1])


	def _prediction_to_transition_probability(
			self,
			initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState
	) -> float:
		predicted_value: float = float(tf.reshape(output, (-1,))[0])
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if final_state.get_market_state().get_state_of(base_currency, quote_currency)[0] == initial_state.get_market_state().get_state_of(base_currency, quote_currency)[0]:
				continue

			if final_state.get_market_state().get_state_of(base_currency, quote_currency)[0] > initial_state.get_market_state().get_state_of(base_currency, quote_currency)[0]:
				return predicted_value

			return 1-predicted_value

	def _get_train_output(self, initial_state: TradeState, action, final_state: TradeState) -> np.ndarray:
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if final_state.get_market_state().get_state_of(base_currency, quote_currency)[0] == initial_state.get_market_state().get_state_of(base_currency, quote_currency)[0]:
				continue

			if final_state.get_market_state().get_state_of(base_currency, quote_currency)[0] > initial_state.get_market_state().get_state_of(base_currency, quote_currency)[0]:
				return np.array([1])

			return np.array([0])

		return np.array([0.5])

	def _get_expected_instant_reward(self, state) -> float:
		return self._get_environment().get_reward(state)

	def _get_possible_states(self, state: TradeState, action: TraderAction) -> List[TradeState]:
		mid_state = self.__simulate_action(state, action)

		states = []

		if action is None or action.action == TraderAction.Action.CLOSE:
			for (base_currency, quote_currency) in state.get_market_state().get_tradable_pairs():
				states += self.__simulate_instrument_change(mid_state, base_currency, quote_currency)
		else:
			states += self.__simulate_instrument_change(mid_state, action.base_currency, action.quote_currency)

		return states

	def __simulate_instrument_change(self, state: TradeState, base_currency: str, quote_currency: str) -> List[TradeState]:
		states = []

		original_value = state.get_market_state().get_state_of(base_currency, quote_currency)

		for j in [-1, 1]:
			new_state = state.__deepcopy__()
			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				np.array(original_value * (1 + j*self.__get_state_change_delta()))
			)
			states.append(new_state)

		return states

	def __simulate_action(self, state: TradeState, action: TraderAction) -> TradeState:  # TODO: SETUP CACHER
		new_state = copy.deepcopy(state)
		new_state.recent_balance = state.get_agent_state().get_balance()
		if action is None:
			return new_state

		if action is None:
			return new_state

		if action.action == TraderAction.Action.CLOSE:
			new_state.get_agent_state().close_trades(action.base_currency, action.quote_currency)
			return new_state

		new_state.get_agent_state().open_trade(
			action,
			state.get_market_state().get_state_of(action.base_currency, action.quote_currency)[0]
		)

		return new_state


class TraderMarkovAgent(MarkovAgent, TraderDNNTransitionAgent):
	
	def __init__(self, *args, **kwargs):
		super(TraderMarkovAgent, self).__init__(*args, **kwargs)


class TraderMonteCarloAgent(MonteCarloAgent, TraderDNNTransitionAgent):

	def __init__(
			self,
			step_time=Config.AGENT_STEP_TIME,
			discount=Config.AGENT_DISCOUNT_FACTOR,
			min_free_memory_percent=Config.MIN_FREE_MEMORY,
			*args, **kwargs
	):
		super(TraderMonteCarloAgent, self).__init__(
			*args,
			discount=discount,
			min_free_memory_percent=min_free_memory_percent,
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
