from core import Config
from lib.network.oanda import Trader


class EnvironmentUtilsProvider:

	@staticmethod
	def provide_trader() -> Trader:
		return Trader(
			Config.OANDA_TOKEN,
			Config.OANDA_TRADING_ACCOUNT_ID,
			timezone=Config.TIMEZONE,
			trading_url=Config.OANDA_TRADING_URL
		)