import unittest

from core.agent.agents.montecarlo_agent.trader_resource_manager import TraderMCResourceManager
from core.di import EnvironmentUtilsProvider


class TraderMCResourceManagerTest(unittest.TestCase):

	def setUp(self):
		self.resource_manager = TraderMCResourceManager(
			trader=EnvironmentUtilsProvider.provide_trader(),
			granularity="M5",
			instrument=("AUD", "USD"),
			delta_multiplier=1
		)

	def test_functionality(self):
		target_time = self.resource_manager.init_resource()
		print(target_time)
