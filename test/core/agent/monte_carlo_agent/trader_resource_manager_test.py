import unittest
import uuid

from core.agent.agents.montecarlo_agent.trader_resource_manager import TraderMCResourceManager
from core.di import EnvironmentUtilsProvider
from lib.rl.agent.mca.resource_manager import DiskResourceManager
from lib.utils.logger import Logger


class TraderMCResourceManagerTest(unittest.TestCase):

	def setUp(self):
		self.resource_manager = TraderMCResourceManager(
			trader=EnvironmentUtilsProvider.provide_trader(),
			granularity="M30",
			instrument=("AUD", "USD"),
			delta_multiplier=10,
			disk_resource_manager=DiskResourceManager(min_remaining_space=0.99)
		)

	@staticmethod
	def __create_dummy_file(size):
		filename = f"{uuid.uuid4().hex}"
		with open(filename, "wb") as f:
			f.write(b'\0' * size * 1024 * 1024)
		Logger.info(f"Created dummy file, {filename} of {size} MB")

	def test_functionality(self):
		target_time = self.resource_manager.init_resource()
		print(target_time)
		self.__create_dummy_file(3*1024)
		self.assertFalse(self.resource_manager.has_resource(target_time))
