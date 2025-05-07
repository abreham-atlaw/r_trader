import os
import typing
from datetime import datetime

from core import Config
from core.utils.misc.sim_trading.setup import SetupManager
from lib.utils.file_storage import FileStorage
from lib.utils.logger import Logger


class ManualRunSetupManager:

	def __init__(
			self,
			fs: FileStorage,
	):
		self.__fs = fs
		self.__setup_manager = SetupManager()

	def __download_model(self, model_path: str) -> str:
		download_path = os.path.abspath(os.path.basename(model_path))
		self.__fs.download(model_path, download_path)
		return download_path

	def setup(
			self,
			start_time: typing.Union[datetime, str],
			model_path: str,
			temperature: float = 1.0,
	):
		if isinstance(start_time, str):
			start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S+00:00")

		Logger.info("Downloading Model...")
		Config.CORE_MODEL_CONFIG.path = self.__download_model(model_path)
		Config.AGENT_MODEL_TEMPERATURE = temperature

		Logger.info("Setting Time...")
		self.__setup_manager.setup(start_time)

		Logger.success("Setup Complete!")
