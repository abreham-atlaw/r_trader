from pymongo import MongoClient

from core import Config
from core.utils.resman import ResourceRepository, MongoResourceRepository
from lib.network.oanda import OandaNetworkClient
from lib.utils.file_storage import FileStorage, PCloudCombinedFileStorage


class ServiceProvider:

	@staticmethod
	def provide_file_storage(path=None) -> FileStorage:
		if path is None:
			path = Config.PCLOUD_FOLDER
		return PCloudCombinedFileStorage(
			tokens=Config.PCLOUD_TOKENS,
			base_path=path
		)

	@staticmethod
	def provide_mongo_client() -> MongoClient:
		return MongoClient(Config.MONGODB_URL)

	@staticmethod
	def provide_resman(category: str) -> ResourceRepository:
		return MongoResourceRepository(category=category, mongo_client=ServiceProvider.provide_mongo_client())

	@staticmethod
	def provide_oanda_client() -> OandaNetworkClient:
		return OandaNetworkClient(
			url=Config.OANDA_TRADING_URL,
			token=Config.OANDA_TOKEN,
			account_id=Config.OANDA_TRADING_ACCOUNT_ID
		)
