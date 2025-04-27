import os.path
import unittest

from core import Config
from core.di import ResearchProvider


class RSSetupManagerTest(unittest.TestCase):

	def setUp(self):

		Config.RunnerStatsBranches.default = Config.RunnerStatsBranches.ma_ews_dynamic_k_stm_it_27
		Config.RunnerStatsLossesBranches.default = Config.RunnerStatsLossesBranches.it_27
		Config.OANDA_SIM_MODEL_IN_PATH = "/Apps/RTrader/maploss/it-27/"

		Config.OANDA_TRADING_ACCOUNT_ID = ""
		Config.OANDA_TRADING_URL = "http://127.0.0.1:8888/api"

		self.setup_manager = ResearchProvider.provide_rs_setup_manager()

	def test_setup(self):
		stat = self.setup_manager.setup()

		self.assertNotEqual(Config.OANDA_TRADING_ACCOUNT_ID, "")
		self.assertEqual(stat.model_name, os.path.basename(Config.CORE_MODEL_CONFIG.path))
		print(stat)