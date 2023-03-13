from typing import *
from abc import ABC

import tensorflow as tf
import numpy as np

from datetime import datetime
import copy
import random
import math


from core import Config
from lib.rl.agent import DNNTransitionAgent, MarkovAgent, MonteCarloAgent, Agent
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, AgentState, ArbitradeTradeState
from core.environment.trade_environment import TradeEnvironment
from .trader_action import TraderAction
from .dnn_models import KerasModelHandler
from .stm import TraderNodeShortTermMemory


class ArbitrageTraderAgent(Agent):

	__STATE_KEY = "ArbitrageTradeState"
	__DIRECTION_ACTION_MAP = {
		-1: TraderAction.Action.SELL,
		1: TraderAction.Action.BUY
	}

	def __init__(
			self,
			zone_size: float = Config.AGENT_ARBITRAGE_ZONE_SIZE,
			base_margin: float = Config.AGENT_ARBITRAGE_BASE_MARGIN,
	):
		super().__init__()
		self.__zone_size = zone_size
		self.__base_margin = base_margin

	def _generate_actions(self, state) -> List[object]:
		pass

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		pass

	def _update_state_action_value(self, initial_state, action, final_state, value):
		pass

	@staticmethod
	def __get_crossing_direction(points: Tuple[float, float], point: float) -> Optional[int]:
		for i, cp in enumerate(points):
			d = 2*(i-0.5)
			if d*point - d*cp > 0 or math.isclose(point, cp):
				return int(d)
		return None

	def __generate_arbitrage_state(self, state: TradeState, instrument: Tuple[str, str]) -> ArbitradeTradeState:
		start_point = state.get_market_state().get_current_price(instrument[0], instrument[1])
		real_zone_size = start_point*self.__zone_size
		spread_cost = state.get_market_state().get_spread_state_of(*instrument)
		spread_zone_size = real_zone_size + 2*spread_cost

		checkpoints = start_point-(real_zone_size/2), start_point+(real_zone_size/2)
		spread_checkpoints = checkpoints[0] - spread_cost, checkpoints[1] + spread_cost
		close_points = spread_checkpoints[0] - spread_zone_size, spread_checkpoints[1] + spread_zone_size

		return ArbitradeTradeState(
			start_point=start_point,
			checkpoints=checkpoints,
			close_points=close_points,
			instrument=instrument
		)

	def __choose_instrument(self, state: TradeState) -> Tuple[str, str]:
		return random.choice(state.get_market_state().get_tradable_pairs())

	def __is_arbitrage_mode(self, state: TradeState) -> bool:
		return state.is_state_attached(self.__STATE_KEY)

	def __get_arbitrage_state(self, state: TradeState) -> ArbitradeTradeState:
		return state.get_attached_state(self.__STATE_KEY)

	def __detach_arbitrage_state(self, state: TradeState):
		state.detach_state(self.__STATE_KEY)

	def __set_arbitrage_state(self, state: TradeState, arbitrage_state: ArbitradeTradeState):
		state.attach_state(self.__STATE_KEY, arbitrage_state)

	def __start_arbitrage_state(self, state: TradeState) -> Optional[TraderAction]:
		instrument = self.__choose_instrument(state)
		arbitrage_state = self.__generate_arbitrage_state(state, self.__choose_instrument(state))
		Logger.info(f"[+]Starting Arbitrage with {arbitrage_state}")
		self.__set_arbitrage_state(state, arbitrage_state)
		if len(state.get_agent_state().get_open_trades(*instrument)) != 0:
			return TraderAction(
				*instrument,
				TraderAction.Action.CLOSE
			)
		return None

	def __on_insufficient_margin(self, state: TradeState) -> Optional[TraderAction]:
		arbitrage_state = self.__get_arbitrage_state(state)
		self.__detach_arbitrage_state(state)
		return TraderAction(
			*arbitrage_state.instrument,
			TraderAction.Action.CLOSE
		)

	def __on_checkpoint(self, state: TradeState, direction: int) -> Optional[TraderAction]:
		arbitrage_state = self.__get_arbitrage_state(state)
		action = self.__DIRECTION_ACTION_MAP[direction]
		trades = state.get_agent_state().get_open_trades()

		margin_size = None

		if arbitrage_state.margin_stage == ArbitradeTradeState.MarginStage.STAGE_ZERO:
			margin_size = self.__base_margin
		elif action == trades[-1].get_trade().action:
			return None
		elif arbitrage_state.margin_stage == ArbitradeTradeState.MarginStage.STAGE_ONE:
			margin_size = 3 * state.get_agent_state().get_open_trades()[-1].get_trade().margin_used

		arbitrage_state.increment_margin_stage()

		if margin_size > state.get_agent_state().get_margin_available():
			Logger.warning("Insufficient Balance to continue sequence...")
			return self.__on_insufficient_margin(state)

		return TraderAction(
			arbitrage_state.instrument[0],
			arbitrage_state.instrument[1],
			action,
			margin_size
		)

	def __on_close_point(self, state: TradeState, direction: int) -> Optional[TraderAction]:
		arbitrage_state = self.__get_arbitrage_state(state)
		action = TraderAction(
			arbitrage_state.instrument[0],
			arbitrage_state.instrument[1],
			TraderAction.Action.CLOSE,
		)
		self.__detach_arbitrage_state(state)
		return action

	def __monitor_arbitrage(self, state: TradeState) -> Optional[TraderAction]:

		arbitrage_state = self.__get_arbitrage_state(state)

		for points, callback in zip(
				[arbitrage_state.checkpoints, arbitrage_state.close_points],
				[self.__on_checkpoint, self.__on_close_point]
		):
			direction = self.__get_crossing_direction(
				points,
				state.get_market_state().get_current_price(*arbitrage_state.instrument)
			)
			if direction is not None:
				action = callback(state, direction)
				if action is not None:
					return action
		return None

	def _policy(self, state, **kwargs):
		if not self.__is_arbitrage_mode(state):
			return self.__start_arbitrage_state(state)
		return self.__monitor_arbitrage(state)


class TraderDNNTransitionAgent(DNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
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
		self.__trade_size_gap = trade_size_gap
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

	def _generate_actions(self, state: TradeState) -> List[Optional[TraderAction]]:
		pairs = state.get_market_state().get_tradable_pairs()

		amounts = [
			(i + 1) * self.__trade_size_gap
			for i in range(int(state.get_agent_state().get_margin_available() // self.__trade_size_gap))
		]

		actions: List[Optional[TraderAction]] = [
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

		actions.append(None)
		return actions

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
			probability_correction=Config.AGENT_PROBABILITY_CORRECTION,
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
			probability_correction=probability_correction,
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
