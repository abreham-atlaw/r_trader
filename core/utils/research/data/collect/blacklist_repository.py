import typing

from pymongo import MongoClient

from core import Config
from core.utils.research.data.collect.runner_stats_repository import RunnerStatsRepository
from lib.utils.cache import Cache


class RSBlacklistRepository:

	def __init__(
			self,
			client: MongoClient,
			db_name: str = "rs-blacklist",
			collection_name: str = "rs-blacklist",
			branch: str = Config.RunnerStatsBranches.default,
	):
		db = client[db_name]
		self._collection = db[f"{collection_name}-branch-{branch}"]
		self.__cached = None

	def __clear_cache(self):
		self.__cached = None

	def __serialize(self, id: str) -> str:
		return {"id": id}

	def __deserialize(self, doc: dict) -> str:
		return doc["id"]

	def add(self, id: str):
		self._collection.insert_one(self.__serialize(id))
		self.__clear_cache()

	def delete(self, id: str):
		self._collection.delete_many(self.__serialize(id))
		self.__clear_cache()

	def is_blacklisted(self, id: str) -> bool:
		return self.get_all().count(id) > 0

	def get_all(self) -> typing.List[str]:
		if self.__cached is not None:
			return self.__cached
		value = list(map(
			self.__deserialize,
			self._collection.find()
		))
		self.__cached = value
		return value

	@staticmethod
	def from_rs_repository(repo: RunnerStatsRepository) -> 'RSBlacklistRepository':
		return RSBlacklistRepository(
			client=repo.client,
			branch=repo.branch
		)
