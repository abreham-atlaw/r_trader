from typing import *

from dataclasses import dataclass

import numpy as np

from lib.rl.agent.mca import Node
from lib.rl.agent.mca.stm import NodeMemoryMatcher, NodeShortTermMemory
from lib.rl.agent.mca.stm.node_memory import NodeMemory
from lib.utils.staterepository import StateRepository
import lib.utils.math as libmath
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, AgentState


@dataclass
class TraderNodeMemory(NodeMemory):
	pass


@dataclass
class AttentionBasedTraderNodeMemory(TraderNodeMemory):
	attention_instrument: Tuple[str, str]


class TraderNodeMemoryMatcher(NodeMemoryMatcher):

	def __init__(
			self,
			threshold: float,
			repository: StateRepository = None,
			average_window=1,
			mean_error: bool = False,
			use_ma_smoothng=False,
			balance_tolerance=None,
			relative_difference=False
	):
		super().__init__(repository=repository, state_matcher=None)
		self.__threshold = threshold
		self.__average_window = average_window
		self.__use_ma_smoothing = use_ma_smoothng
		self.__mean_error = mean_error
		self.__balance_tolerance = balance_tolerance
		if balance_tolerance is None:
			self.__balance_tolerance = 0.01
		self.__relative_difference = relative_difference

	def construct_memory(self, node: Node) -> TraderNodeMemory:
		return TraderNodeMemory(
			node=node
		)

	@staticmethod
	def __get_relative_difference(state: np.ndarray) -> np.ndarray:
		return state[1:] - state[:-1]

	def __error(self, state0: np.ndarray, state1: np.ndarray) -> float:
		if self.__relative_difference:
			state0, state1 = [
				self.__get_relative_difference(s)
				for s in [state0, state1]
			]

		error = np.abs(state0 - state1)
		if self.__mean_error:
			error = np.mean(error)
		else:
			error = np.sum(error)
		return float(error)

	def _compare_instrument_history(self, state0: np.ndarray, state1: np.ndarray) -> float:
		if self.__use_ma_smoothing:
			state0, state1 = [
				libmath.moving_average(state, self.__average_window)
				for state in [state0, state1]
			]
		return self.__error(
			state0,
			state1
		)

	@staticmethod
	def __compare_trades(trade0: AgentState.OpenTrade, trade1: AgentState.OpenTrade) -> bool:
		return trade0.get_trade() == trade1.get_trade()

	def __compare_agent_states(self, state0: AgentState, state1: AgentState) -> bool:
		return \
			np.isclose(state0.get_balance(), state1.get_balance(), atol=self.__balance_tolerance) and \
			len(state0.get_open_trades()) == len(state1.get_open_trades()) and \
			np.all([self.__compare_trades(
					trade0,
					trade1
				) for trade0, trade1 in zip(*[
						sorted(state.get_open_trades(), key=lambda t: t.get_enter_value())
						for state in [state0, state1]
					]
				)
			])

	def __compare_discrete_values(self, cue: TradeState, memory: TradeState) -> bool:
		return self.__compare_agent_states(cue.get_agent_state(), memory.get_agent_state())

	def _get_comparable_instruments(self, memory0: TraderNodeMemory, memory1: TraderNodeMemory) -> List[Tuple[str, str]]:
		state: TradeState = self.get_repository().retrieve(memory0.node.id)
		return state.get_market_state().get_tradable_pairs()

	def __evaluate_stochastic_value(self, memory0: TraderNodeMemory, memory1: TraderNodeMemory) -> float:

		return float(
			np.mean([
				self._compare_instrument_history(
					state0=self.get_repository().retrieve(memory0.node.id).get_market_state().get_state_of(
						base_currency=base_currency,
						quote_currency=quote_currency
					),
					state1=self.get_repository().retrieve(memory1.node.id).get_market_state().get_state_of(
						base_currency=base_currency,
						quote_currency=quote_currency
					)
				)
				for base_currency, quote_currency in self._get_comparable_instruments(memory0, memory1)
			])
		)

	def __compare_stochastic_values(self, memory0: TraderNodeMemory, memory1: TraderNodeMemory) -> bool:
		sv = self.__evaluate_stochastic_value(memory0, memory1)
		return sv < self.__threshold

	def _compare_memories(self, memory0: TraderNodeMemory, memory1: TraderNodeMemory) -> bool:

		return \
			self.__compare_discrete_values(
				self.get_repository().retrieve(memory0.node.id),
				self.get_repository().retrieve(memory1.node.id)
			) and \
			self.__compare_stochastic_values(
				memory0,
				memory1
			)

	def is_match(self, cue: 'Node', memory: TraderNodeMemory) -> bool:
		return self._compare_memories(
			self.construct_memory(cue),
			memory
		)


class AttentionBasedTraderNodeMemoryMatcher(TraderNodeMemoryMatcher):

	@staticmethod
	def __find_changed_instrument(state0: TradeState, state1: TradeState) -> Optional[Tuple[str, str]]:
		for base_currency, quote_currency in state0.get_market_state().get_tradable_pairs():
			if not np.all(
					state0.get_market_state().get_state_of(
						base_currency,
						quote_currency
					) == state1.get_market_state().get_state_of(
						base_currency,
						quote_currency
					)
			):
				return base_currency, quote_currency

		return None

	def construct_memory(self, node: Node) -> TraderNodeMemory:
		attention_instrument = None
		if node.parent is not None:
			attention_instrument = self.__find_changed_instrument(
				self.get_repository().retrieve(node.id),
				self.get_repository().retrieve(node.parent.parent.id)
			)

		return AttentionBasedTraderNodeMemory(
			node=node,
			attention_instrument=attention_instrument
		)

	def _get_comparable_instruments(
			self,
			memory0: AttentionBasedTraderNodeMemory,
			memory1: AttentionBasedTraderNodeMemory
	) -> List[Tuple[str, str]]:
		for instrument in [memory0.attention_instrument, memory1.attention_instrument]:
			if instrument is not None:
				return [instrument]

		raise Exception("No Instrument found on memory")

	def _compare_memories(self, memory0: AttentionBasedTraderNodeMemory, memory1: AttentionBasedTraderNodeMemory) -> bool:
		return \
				not (
						memory0.attention_instrument != memory1.attention_instrument and
						not (memory0.attention_instrument is None or memory1 .attention_instrument is None)
				) and \
				super()._compare_memories(memory0, memory1)


class TraderNodeShortTermMemory(NodeShortTermMemory):

	def __init__(
			self,
			size: int,
			matcher: TraderNodeMemoryMatcher,
	):
		super().__init__(
			size,
			matcher
		)

	def _import_memory(self, node: Node) -> TraderNodeMemory:
		return self.get_matcher().construct_memory(node)

	def _sort_memory(self, memories: List[TraderNodeMemory], recall_index=None) -> List[object]:
		return sorted(memories, key=lambda memory: memory.node.weight, reverse=True)
