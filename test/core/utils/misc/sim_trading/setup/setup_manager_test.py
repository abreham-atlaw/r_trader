import unittest
from datetime import datetime

from core import Config
from core.utils.misc.sim_trading.setup import SetupManager


class SetupManagerTest(unittest.TestCase):

	def setUp(self):
		Config.OANDA_TRADING_ACCOUNT_ID = ""
		Config.OANDA_TRADING_URL = "http://127.0.0.1:8888/api"
		self.manager = SetupManager()

	def test_functionality(self):

		summary = self.manager.setup(
			start_time=datetime.strptime("2024-02-28 13:07:00+00:00", "%Y-%m-%d %H:%M:%S%z")
		)
		self.assertNotEqual(Config.OANDA_TRADING_ACCOUNT_ID, "")
		print(Config.OANDA_TRADING_ACCOUNT_ID)
		print(summary)
