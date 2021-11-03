from typing import *

import numpy as np

import copy

from lib.rl.agent import DNNAgent
from .trade_transition_model import TransitionModel
from core.environment.trade_state import TradeState
from .trader_action import TraderAction
from core.environment.trade_environment import TradeEnvironment


class TraderAgent(DNNAgent):

	def __init__(self, *args, trade_size_gap=5, state_change_delta=0.01, **kwargs):
		super().__init__(*args, episodic=False, depth=100, **kwargs)
		self.__trade_size_gap = trade_size_gap
		self.__state_change_delta = state_change_delta
		self.environment: TradeEnvironment
		self.set_transition_model(
			TransitionModel()
		)

	def _state_action_to_model_input(self, state: TradeState, action: TraderAction, final_state: TradeState) -> np.ndarray:
		return state.get_market_state().get_state_of(action.base_currency, action.quote_currency)

	def _prediction_to_transition_probability(
			self,
			initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState
	) -> float:
		predicted_value: float = output.reshape((-1,))[0]
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

	def __simulate_action(self, state: TradeState, action: TraderAction) -> TradeState:
		new_state = copy.deepcopy(state)
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

	def _get_possible_states(self, state: TradeState, action: TraderAction) -> List[TradeState]:
		mid_state = self.__simulate_action(state, action)

		states = []
		for (base_currency, quote_currency) in state.get_market_state().get_tradable_pairs():
			original_value = state.get_market_state().get_state_of(base_currency, quote_currency)[0]

			for j in [-1, 1]:
				new_state = copy.deepcopy(mid_state)
				new_state.get_market_state().update_state_of(
					base_currency,
					quote_currency,
					np.array([original_value * (1 + j*self.__state_change_delta)])
				)
				new_state.get_market_state().update_state_of(
					quote_currency,
					base_currency,
					1/new_state.get_market_state().get_state_of(base_currency, quote_currency)
				)

				states.append(new_state)

		return states
