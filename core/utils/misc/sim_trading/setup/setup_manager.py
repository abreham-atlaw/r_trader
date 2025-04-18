from datetime import datetime

from core import Config
from core.di import ServiceProvider
from core.utils.misc.sim_trading.setup.requests import CreateAccountRequest
from lib.network.oanda import OandaNetworkClient
from lib.network.oanda.data.models import AccountSummary


class SetupManager:

	def __init__(self):
		self.__client = ServiceProvider.provide_oanda_client()

	def setup(
		self,
		start_time: datetime
	):
		summary: AccountSummary = self.__client.execute(CreateAccountRequest(
			start_time=start_time,
			delta_multiplier=Config.OANDA_SIM_DELTA_MULTIPLIER,
			margin_rate=Config.OANDA_SIM_MARGIN_RATE,
			alias=Config.OANDA_SIM_ALIAS,
			balance=Config.OANDA_SIM_BALANCE
		))

		Config.OANDA_TRADING_ACCOUNT_ID = summary.id
		return summary
