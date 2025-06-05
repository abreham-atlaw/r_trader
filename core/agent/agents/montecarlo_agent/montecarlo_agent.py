from abc import ABC

from datetime import datetime

from core import Config
from core.di import AgentUtilsProvider
from lib.rl.agent import MonteCarloAgent
from core.agent.agents.dnn_transition_agent import TraderDNNTransitionAgent
from lib.rl.agent.mca.resource_manager import TimeMCResourceManager
from .stm import TraderNodeShortTermMemory, TraderNodeMemoryMatcher


class TraderMonteCarloAgent(MonteCarloAgent, ABC):

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
			stm_use_ma_smoothing=Config.AGENT_STM_USE_MA_SMOOTHING,
			stm_mean_error=Config.AGENT_STM_MEAN_ERROR,
			stm_attention_mode=Config.AGENT_STM_ATTENTION_MODE,
			probability_correction=Config.AGENT_PROBABILITY_CORRECTION,
			min_probability=Config.AGENT_MIN_PROBABILITY,
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
				matcher=TraderNodeMemoryMatcher(
					threshold=stm_threshold,
					use_ma_smoothng=stm_use_ma_smoothing,
					mean_error=stm_mean_error,
					balance_tolerance=stm_balance_tolerance,
					average_window=stm_average_window,
				)
			),
			probability_correction=probability_correction,
			min_probability=min_probability,
			resource_manager=AgentUtilsProvider.provide_resource_manager(),
			**kwargs
		)

	def _get_state_node_instant_value(self, state_node: 'MonteCarloAgent.Node') -> float:
		return self._get_environment().get_reward(state_node.state) - self._get_environment().get_reward(
			state_node.parent.parent.state)
