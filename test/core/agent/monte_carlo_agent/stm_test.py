import unittest

import numpy as np

from core import Config
from core.agent.agents.montecarlo_agent.stm import TraderNodeMemoryMatcher
from core.di import AgentUtilsProvider
from core.environment.trade_state import MarketState, AgentState, TradeState
from lib.rl.agent import Node


class TraderNodeMemoryMatcherTest(unittest.TestCase):

	def setUp(self):
		self.bounds = np.array(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND)
		self.bounds_diff = self.bounds[1:] - self.bounds[:-1]
		self.state_repository = AgentUtilsProvider.provide_state_repository()
		self.threshold = np.min(self.bounds_diff)/2
		self.noise = self.threshold * 0.01
		self.matcher = TraderNodeMemoryMatcher(
			threshold=self.threshold,
			repository=self.state_repository,
			relative_difference=True
		)



