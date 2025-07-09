import unittest

import os

from core import Config
from core.di import ResearchProvider
from core.utils.research.data.collect.sim_setup.rt_rs_setup_manager import RealTimeRSSetupManager
from lib.utils.logger import Logger


class RealTimeRSSetupManagerTest(unittest.TestCase):

	def setUp(self):
		Config.RunnerStatsBranches.default = Config.RunnerStatsBranches.it_23_0
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_23
		Config.OANDA_SIM_MODEL_IN_PATH = "/Apps/RTrader/maploss/it-23/"

		Config.OANDA_TRADING_ACCOUNT_ID = ""
		Config.OANDA_TRADING_URL = "https://api-fxpractice.oanda.com/v3"

		Config.UPDATE_SAVE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/train"

		self.setup_manager: RealTimeRSSetupManager = ResearchProvider.provide_rt_rs_setup_manager()

	def test_setup(self):
		stat = self.setup_manager.setup()

		self.assertNotEqual(Config.OANDA_TRADING_ACCOUNT_ID, "")
		self.assertEqual(stat.model_name, os.path.basename(Config.CORE_MODEL_CONFIG.path))
		Logger.info(f"Allocated Account: {Config.OANDA_TRADING_ACCOUNT_ID}")

	def test_setup_and_finish(self):
		stat = self.setup_manager.setup()

		self.setup_manager.finish(stat, 1.5)