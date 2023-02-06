import typing
from abc import ABC, abstractmethod

from pymongo import MongoClient


class MetricRepository(ABC):

	@staticmethod
	def __construct_container(metrics: typing.List['Trainer.Metric']) -> 'Trainer.MetricsContainer':
		from .trainer import Trainer
		container = Trainer.MetricsContainer()
		for m in metrics:
			container.add_metric(m)
		return container

	@abstractmethod
	def write_metric(self, metric: 'Trainer.Metric'):
		pass

	@abstractmethod
	def exists(self, metric: 'Trainer.Metric') -> bool:
		pass

	@abstractmethod
	def _get_all(self) -> typing.List['Trainer.Metric']:
		pass

	def get_all(self) -> 'Trainer.MetricContainer':
		return self.__construct_container(self._get_all())


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

	@staticmethod
	def __construct_metric(json_):
		from .trainer import Trainer
		metric = Trainer.Metric(None, None, None, None, None)
		for k in metric.__dict__.keys():
			metric.__dict__[k] = json_[k]
		return metric

	def _get_all(self) -> typing.List['Trainer.Metric']:
		return [self.__construct_metric(j) for j in list(self._collection.find())]

