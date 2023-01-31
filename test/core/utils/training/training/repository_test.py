import typing

import unittest

import pymongo.errors

from core.utils.training.training.repository import MongoDBMetricRepository
from core.utils.training.training.trainer import Trainer
from core import Config


class MongoDBMetricRepositoryTest(unittest.TestCase):

	__METRIC = Trainer.Metric(1, 2, 3, 4, (3, 4))

	def setUp(self) -> None:
		self.__repository = MongoDBMetricRepository(Config.MONGODB_URL, "test")

	def test_write_metric(self):
		self.__repository.write_metric(self.__METRIC)

	def test_exists(self):
		try:
			self.__repository.write_metric(self.__METRIC)
		except pymongo.errors.DuplicateKeyError:
			print("Metric Already Written")
		self.assertTrue(
			self.__repository.exists(self.__METRIC)
		)
