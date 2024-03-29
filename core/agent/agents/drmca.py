import random
import typing
from abc import ABC

import numpy as np

from core import Config
from core.agent.agents.dnn_transition_agent import TraderDNNTransitionAgent
from core.agent.trader_action import TraderAction
from core.environment.trade_state import TradeState
from lib.rl.agent.drmca import DeepReinforcementMonteCarloAgent
from lib.rl.agent.dta import Model, TorchModel
from lib.rl.environment import ModelBasedState


class TraderDeepReinforcementMonteCarloAgent(DeepReinforcementMonteCarloAgent, TraderDNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			batch_size=Config.UPDATE_EXPORT_BATCH_SIZE,
			train=Config.UPDATE_TRAIN,
			save_path=Config.UPDATE_SAVE_PATH,
			**kwargs
	):
		super().__init__(
			*args,
			batch_size=batch_size,
			train=train,
			save_path=save_path,
			**kwargs
		)

	def _init_model(self) -> Model:
		return TorchModel.load(Config.CORE_MODEL_CONFIG.path)

	@property
	def _transition_model(self) -> Model:
		return self._model

	@staticmethod
	def __encode_action(state: TradeState, action: typing.Optional[TraderAction]) -> np.ndarray:
		encoded = np.zeros((4,))
		if action is None:
			return encoded
		encoded[action.action] = 1
		if action.action != TraderAction.Action.CLOSE:
			encoded[3] = state.agent_state.calc_required_margin(action.units, action.base_currency, action.quote_currency)/state.agent_state.get_margin_available()
		return encoded

	def __prepare_model_input(
			self,
			state: TradeState,
			action: typing.Optional[TraderAction],
			target_instrument: typing.Tuple[str, str]
	) -> np.ndarray:
		time_series = state.get_market_state().get_state_of(target_instrument[0], target_instrument[1])
		encoded_action = self.__encode_action(state, action)
		return np.concatenate((time_series, encoded_action), axis=0)

	def _prepare_single_dta_input(self, state: TradeState, action: TraderAction, final_state: TradeState) -> np.ndarray:
		return self.__prepare_model_input(state, action, self._get_target_instrument(state, action, final_state))

	def _prepare_dra_input(self, state: TradeState, action: TraderAction) -> np.ndarray:
		if action is None:
			instrument = random.choice(state.get_market_state().get_tradable_pairs())
		else:
			instrument = action.base_currency, action.quote_currency
		return self.__prepare_model_input(state, action, instrument)

	@staticmethod
	def _parse_model_output(output: np.ndarray) -> typing.Tuple[np.ndarray, float]:
		probability_distribution = output[:-1]
		value = output[-1]
		return probability_distribution, value

	def _prepare_dra_output(self, state: TradeState, action: TraderAction, output: np.ndarray) -> float:
		_, value = self._parse_model_output(output)
		return value

	def _single_prediction_to_transition_probability_bound_mode(
			self,
			initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState
	) -> float:
		output, _ = self._parse_model_output(output)
		return super()._single_prediction_to_transition_probability_bound_mode(initial_state, output, final_state)

	def _prepare_dra_train_output(
			self,
			state: TradeState,
			action: TraderAction,
			final_state: TradeState,
			value: float
	) -> np.ndarray:
		instrument = self._get_target_instrument(state, action, final_state)
		percentage = final_state.market_state.get_current_price(*instrument)/state.market_state.get_current_price(*instrument)
		bound_idx = self._find_gap_index(percentage)
		output = np.zeros(len(self._state_change_delta_bounds)+1)
		output[bound_idx] = 1
		output[-1] = value
		return output

	def _update_state_action_value(self, initial_state: ModelBasedState, action, final_state: ModelBasedState, value):
		DeepReinforcementMonteCarloAgent._update_state_action_value(self, initial_state, action, final_state, value)
