import typing

import unittest

from core.utils.training.training.continuoustrainer.repository import PCloudTrainerRepository


class PCloudTrainerRepositoryTest(unittest.TestCase):

	ID = "28"
	URLS = ("https://example.com/url0", "https://example.com/url1")
	EPOCH = 4

	def test_functionality(self):
		repository = PCloudTrainerRepository("/Apps/RTrader")
		repository.update_checkpoint(
			self.ID,
			self.URLS,
			self.EPOCH
		)

		urls, epoch = repository.get_checkpoint(self.ID)
		self.assertTupleEqual(self.URLS, urls)
		self.assertEqual(self.EPOCH, epoch)