import typing
from dataclasses import dataclass

from pymongo import MongoClient

from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_serializer import RunnerStatsSerializer


class RunnerStatsRepository:

	def __init__(
			self,
			client: MongoClient,
			db_name: str = "runner_stats",
			collection_name: str = "runner_stats"
	):
		db = client[db_name]
		self._collection = db[collection_name]
		self.__serializer = RunnerStatsSerializer()

	def store(self, stats: RunnerStats):
		old_stats = self.retrieve(stats.id)
		if old_stats is None:
			self._collection.insert_one(
				stats.__dict__
			)
			return
		if stats.profit == 0:
			stats.profit = old_stats.profit
		old_stats.duration += stats.duration
		old_stats.profit = stats.profit
		self._collection.update_one(
			{"id": old_stats.id},
			{"$set": old_stats.__dict__},
			upsert=True
		)

	def retrieve(self, id: str) -> typing.Optional[RunnerStats]:
		doc = self._collection.find_one({"id": id})
		if doc:
			return self.__serializer.deserialize(doc)
		else:
			return None

	def retrieve_all(self) -> typing.List[RunnerStats]:
		docs = self._collection.find()
		return self.__serializer.deserialize_many(docs)

	def exists(self, id):
		return self.retrieve(id) is not None

	def retrieve_by_loss_complete(self) -> typing.List[RunnerStats]:
		return [
			stat
			for stat in self.retrieve_all()
			if 0.0 not in stat.model_losses
		]
