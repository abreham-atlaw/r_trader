import unittest

from core import Config
from core.di import ServiceProvider
from core.utils.resman.data.repositories.resource_repository import ResourceUnavailableException


class MongoResourceRepository(unittest.TestCase):

	def setUp(self):
		self.repository = ServiceProvider.provide_resman(Config.ResourceCategories.TEST_RESOURCE)


	def test_create(self) :
		for i in range(20):
			self.repository.create(str(i))

	def test_allocate(self):

		allocated = []
		exhausted = False

		while not exhausted:
			try:
				allocated.append(self.repository.allocate().id)
			except ResourceUnavailableException:
				exhausted = True
		print(allocated)
		self.assertEqual(len(allocated), len(set(allocated)))

	def test_release(self):

		for resource in self.repository.get_locked():
			self.repository.release(resource)

		self.assertEqual(len(self.repository.get_locked()), 0)
