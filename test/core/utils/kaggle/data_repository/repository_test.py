import unittest

import os

from core.utils.kaggle.data_repository import KaggleDataRepository


class KaggleDataRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.output_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/test"
		self.repository = KaggleDataRepository(
			output_path=self.output_path
		)

	def test_download(self):
		self.repository.download("abrehamatlaw0/mock-notebook-0")
		os.system(f"tree \"{self.output_path}\"")

