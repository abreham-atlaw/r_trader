import unittest

import json

from core import Config
from core.di import ServiceProvider
from core.utils.resman.data.repositories.resource_repository import ResourceUnavailableException


class MongoResourceRepository(unittest.TestCase):

	def setUp(self):
		self.repository = ServiceProvider.provide_resman(Config.ResourceCategories.OANDA_SIM_ACCOUNTS)

	def test_create(self):
		with open("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/accounts/accounts.json") as f:
			ids = json.load(f)

		for id in ids:
			self.repository.create(id)

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

		locked = self.repository.get_locked()

		print(f"Releasing {len(locked)} Resources")

		for resource in locked:
			self.repository.release(resource)

		self.assertEqual(len(self.repository.get_locked()), 0)
