import unittest

import numpy as np

from core.di import ResearchProvider
from core.utils.research.training.data.repositories.ims_repository import IMSRepository


class IMSRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.model_name = "test-model"
		self.repository: IMSRepository = ResearchProvider.provide_ims_repository(
			model_name=self.model_name,
			label="test_value"
		)

	def test_store(self):
		for i in range(10):
			self.repository.store(
				value=np.random.random((10, 100)),
				epoch=0,
				batch=1
			)
		self.repository.sync()

	def test_retrieve(self):
		df = self.repository.retrieve()
		print(df)

	def test_retrieve_X(self):
		repository: IMSRepository = ResearchProvider.provide_ims_repository(
			model_name=self.model_name,
			label="X"
		)

		df = repository.retrieve()
		print(df)

