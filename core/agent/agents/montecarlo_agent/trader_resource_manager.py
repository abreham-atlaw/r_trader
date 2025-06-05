import typing
from datetime import datetime, timedelta

from lib.network.oanda import Trader
from lib.rl.agent.mca.resource_manager import MCResourceManager
from lib.utils.logger import Logger


class TraderMCResourceManager(MCResourceManager):

	def __init__(
			self,
			trader: Trader,
			granularity: str,
			instrument: typing.Tuple[str, str],
			delta_multiplier: float = 1,
	):
		self.__trader = trader
		self.__gran = granularity
		self.__gran_value = self.__trader.get_granularity_seconds(self.__gran)
		self.__instrument = instrument
		self.__delta_multiplier = delta_multiplier

	def __round_time(self, date: datetime) -> datetime:
		minute_gran = self.__gran_value // 60
		return date.replace(minute=(date.minute // minute_gran) * minute_gran, second=0, microsecond=0)  # TODO: THIS ONLY SUPPORT MINUTE BASED GRANS

	def init_resource(self) -> datetime:
		current_time = self.__trader.get_current_time(self.__instrument).replace(tzinfo=None)
		Logger.info(f"[TraderMCResourceManager] Current Time: {current_time}")
		target_time = self.__round_time(
			current_time + timedelta(seconds=self.__gran_value / self.__delta_multiplier)
		)
		Logger.info(f"[TraderMCResourceManager] Target Time: {target_time}")
		return target_time

	def has_resource(self, resource: datetime) -> bool:
		return (datetime.now() - resource).total_seconds() <= 0
