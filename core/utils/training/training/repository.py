import typing
from abc import ABC, abstractmethod

from pymongo import MongoClient


class MetricRepository(ABC):

	@abstractmethod
	def write_metric(self, metric: 'Trainer.Metric'):
		pass

	@abstractmethod
	def exists(self, metric: 'Trainer.Metric') -> bool:
		pass


class MongoDBMetricRepository(MetricRepository):

	def __init__(self, url: str, models_id: str, db_name="metrics"):
		self.__client = MongoClient(url)
		self.__db = self.__client[db_name]
		self._collection = self.__db[models_id]

	def write_metric(self, metric: 'Trainer.Metric'):
		self._collection.insert_one(metric.__dict__)

	def exists(self, metric: 'Trainer.Metric') -> bool:
		response = self._collection.find_one(metric.__dict__)
		return response is not None

