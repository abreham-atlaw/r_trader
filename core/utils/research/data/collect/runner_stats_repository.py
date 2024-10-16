import random
import typing
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pymongo import MongoClient

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.collect.runner_stats import RunnerStats
from core.utils.research.data.collect.runner_stats_serializer import RunnerStatsSerializer


class RunnerStatsRepository:

	def __init__(
			self,
			client: MongoClient,
			db_name: str = "runner_stats",
			collection_name: str = "runner_stats",
			select_weight: float = 0.5,
			max_loss: float = 6,
			model_name_key: str = "",
			population_size: int = 160
	):
		db = client[db_name]
		self._collection = db[collection_name]
		self.__serializer = RunnerStatsSerializer()
		self.__resman = ServiceProvider.provide_resman(Config.ResourceCategories.RUNNER_STAT)
		self.__select_weight = select_weight
		self.__max_loss = max_loss
		self.__model_name_key = model_name_key
		self.__population_size = population_size

	def __get_select_sort_field(self, stat: RunnerStats) -> float:
		return stat.duration

	def __filter_select(self, stats: typing.List[RunnerStats]):
		selected = list(filter(
			lambda stat: stat.model_losses[0] <= self.__max_loss and self.__model_name_key in stat.model_name,
			stats
		))
		if self.__population_size is not None:
			selected = selected[:self.__population_size]
		return selected

	def store(self, stats: RunnerStats):
		old_stats = self.retrieve(stats.id)
		if old_stats is None:
			self._collection.insert_one(
				stats.__dict__
			)
			return
		self._collection.update_one(
			{"id": stats.id},
			{"$set": stats.__dict__},
			upsert=True
		)

	def remove(self, id: str):
		self._collection.delete_one({"id": id})

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

	def retrieve_non_locked(self) -> typing.List[RunnerStats]:
		return [
			stat
			for stat in self.retrieve_all()
			if not self.__resman.is_locked_by_id(stat.id)
		]

	def allocate_for_runlive(self, allow_locked: bool = False) -> RunnerStats:
		if allow_locked:
			pool = self.retrieve_all()
		else:
			pool = self.retrieve_non_locked()

		pool = self.__filter_select(pool)

		sorted_pool = sorted(
			pool,
			key=lambda stat: self.__get_select_sort_field(stat) + (self.__select_weight * np.mean([self.__get_select_sort_field(stat) for stat in pool]) * random.random())
		)
		if len(sorted_pool) == 0:
			if not allow_locked:
				raise Exception("No stats available")
			print("No stats available. Allocating for locked...")
			return self.allocate_for_runlive(allow_locked=True)
		selected = sorted_pool[0]
		selected.add_session_timestamp(datetime.now())
		self.__resman.lock_by_id(selected.id)
		return selected

	def finish_session(self, instance: RunnerStats, profit: float):
		duration = (datetime.now() - instance.session_timestamps[-1]).total_seconds()
		instance.add_profit(profit)
		instance.add_duration(duration)
		self.store(instance)
		self.__resman.unlock_by_id(instance.id)

	def delete(self, id: str):
		self._collection.delete_one({"id": id})

	def retrieve_valid(self):
		return self.__filter_select(self.retrieve_all())
