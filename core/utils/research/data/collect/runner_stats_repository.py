import typing
from dataclasses import dataclass

from pymongo import MongoClient


@dataclass
class RunnerStats:
	id: str
	model_url: str
	value: float = 0.0
	duration: float = 0.0
	accuracy: float = 0.0


class RunnerStatsRepository:

	def __init__(
			self,
			client: MongoClient,
			db_name: str = "runner_stats",
			collection_name: str = "runner_stats"
	):
		db = client[db_name]
		self._collection = db[collection_name]

	def store(self, stats: RunnerStats):
		old_stats = self.retrieve(stats.id)
		if old_stats is None:
			self._collection.insert_one(
				stats.__dict__
			)
			return
		old_stats += stats.duration
		old_stats.value = stats.value
		self._collection.update_one(
			{"id": old_stats.id},
			{"$set": old_stats.__dict__},
			upsert=True
		)

	def retrieve(self, id: str) -> typing.Optional[RunnerStats]:
		doc = self._collection.find_one({"id": id})
		if doc:
			doc.pop('_id')
			return RunnerStats(**doc)
		else:
			return None
