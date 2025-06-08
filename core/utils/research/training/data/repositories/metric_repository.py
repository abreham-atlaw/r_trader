import typing
from abc import ABC, abstractmethod

from pymongo import MongoClient
from pymongo.errors import AutoReconnect

from core.utils.research.training.data.metric import Metric, MetricsContainer
from lib.utils.decorators import retry


class MetricRepository(ABC):

	@staticmethod
	def __construct_container(metrics: typing.List[Metric]) -> MetricsContainer:
		container = MetricsContainer()
		for m in metrics:
			container.add_metric(m)
		return container

	@abstractmethod
	def write_metric(self, metric: Metric):
		pass

	@abstractmethod
	def exists(self, metric: Metric) -> bool:
		pass

	@abstractmethod
	def _get_all(self) -> typing.List[Metric]:
		pass

	def get_all(self) -> MetricsContainer:
		return self.__construct_container(self._get_all())


class MongoDBMetricRepository(MetricRepository):

	def __init__(self, url: str, models_id: str, db_name="metrics"):
		self.__client = MongoClient(url)
		self.__db = self.__client[db_name]
		self._collection = self.__db[models_id]

	def write_metric(self, metric: Metric):
		self._collection.insert_one(metric.__dict__)

	def exists(self, metric: Metric) -> bool:
		response = self._collection.find_one(metric.__dict__)
		return response is not None

	@staticmethod
	def __construct_metric(json_):
		metric = Metric(None, None, None, None, None)
		for k in metric.__dict__.keys():
			metric.__dict__[k] = json_[k]
		return metric

	@retry(exception_cls=AutoReconnect, patience=10, sleep_timer=5)
	def _get_all(self) -> typing.List[Metric]:
		return [self.__construct_metric(j) for j in list(self._collection.find())]

