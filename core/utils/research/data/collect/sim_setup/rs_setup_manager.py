import os
import typing
from datetime import datetime

from core import Config
from core.utils.misc.sim_trading.setup import SetupManager
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from core.utils.research.data.collect.sim_setup.times_repository import TimesRepository
from lib.utils.file_storage import FileStorage
from lib.utils.logger import Logger


class RSSetupManager:

	def __init__(
			self,
			times_repo: TimesRepository,
			rs_repo: RunnerStatsRepository,
			fs: FileStorage
	):
		self.__times_repo = times_repo
		self.__rs_repo = rs_repo
		self.__fs = fs
		self.__setup_manager = SetupManager()

	def __serialize_time(self, time: datetime):
		return time.strftime("%Y-%m-%d %H:%M:%S+00:00")

	def _allocate_time(self, stat: RunnerStats) -> datetime:
		return min(
			self.__times_repo.retrieve_all(),
			key=lambda t: stat.simulated_timestamps.count(self.__serialize_time(t))
		)

	def setup(self):
		Logger.info(f"Allocating Runner Stat...")
		stat = self.__rs_repo.allocate_for_runlive()
		Logger.info(f"Allocated Runner Stat: {stat}")

		Logger.info(f"Downloading {stat.model_name}...")
		self.__fs.download(stat.model_name, os.path.abspath(stat.model_name))

		Config.CORE_MODEL_CONFIG.path = os.path.abspath(stat.model_name)
		Config.AGENT_MODEL_TEMPERATURE = stat.temperature

		Logger.info(f"Allocating Time...")
		start_time = self._allocate_time(stat)
		Logger.info(f"Allocated Start Time: {start_time}")
		stat.simulated_timestamps.append(self.__serialize_time(start_time))

		Logger.info(f"Setting up Simulation Trading...")
		self.__setup_manager.setup(start_time)

		Logger.success(f"Setup Complete!")

		return stat
