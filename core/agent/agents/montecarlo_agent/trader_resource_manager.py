import time
import typing
from datetime import datetime, timedelta

from lib.network.oanda import Trader
from lib.rl.agent.mca.resource_manager import MCResourceManager, DiskResourceManager
from lib.utils.logger import Logger


class TraderMCResourceManager(MCResourceManager):

	def __init__(
			self,
			trader: Trader,
			granularity: str,
			instrument: typing.Tuple[str, str],
			delta_multiplier: float = 1,
			disk_resource_manager: DiskResourceManager = None
	):
		self.__trader = trader
		self.__gran = granularity
		self.__gran_value = self.__trader.get_granularity_seconds(self.__gran)
		self.__instrument = instrument
		self.__delta_multiplier = delta_multiplier
		self.__disk_resource_manager = disk_resource_manager if disk_resource_manager else DiskResourceManager()

	def __round_time(self, date: datetime) -> datetime:
		minute_gran = self.__gran_value // 60
		return date.replace(minute=(date.minute // minute_gran) * minute_gran, second=0, microsecond=0)  # TODO: THIS ONLY SUPPORT MINUTE BASED GRANS

	def init_resource(self) -> typing.Tuple[datetime, float]:
		current_local_time = datetime.now()
		current_time = self.__trader.get_current_time(self.__instrument).replace(tzinfo=None)
		Logger.info(f"[TraderMCResourceManager] Current Time: {current_time}({current_local_time})")
		target_time = self.__round_time(
			current_time + timedelta(seconds=self.__gran_value)
		)
		target_local_time = current_local_time + ((target_time - current_time)/self.__delta_multiplier)
		Logger.info(f"[TraderMCResourceManager] Target Time: {target_time}({target_local_time})")

		disk_resource = self.__disk_resource_manager.init_resource()
		return target_local_time, disk_resource

	def has_resource(self, resource: typing.Tuple[datetime, float]) -> bool:
		resource, disk_resource = resource
		if self.__disk_resource_manager.has_resource(disk_resource):
			return (datetime.now() - resource).total_seconds() <= 0

		remaining_time = (resource - datetime.now()).total_seconds()
		Logger.warning(f"Out of disk space. Sleeping {remaining_time} seconds.")
		time.sleep(remaining_time)

		return False
