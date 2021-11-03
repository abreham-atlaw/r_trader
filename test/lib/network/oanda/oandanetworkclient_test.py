from typing import *

import unittest

from lib.network.oanda.requests import AccountSummary, AccountSummaryRequest
from lib.network.oanda.data.models import AccountSummary
from lib.network.oanda.OandaNetworkClient import OandaNetworkClient
from core import Config


class OandaNetworkClientTest(unittest.TestCase):

	def setUp(self) -> None:
		self.client = OandaNetworkClient(
			Config.OANDA_TRADING_URL,
			Config.OANDA_TOKEN,
			Config.OANDA_TEST_ACCOUNT_ID
		)
	

	def test_execute(self):
		response: AccountSummary = self.client.execute(AccountSummaryRequest())
		self.assertIsInstance(response, AccountSummary)


if __name__ == "__main__":
	unittest.main()
