import unittest

import os

from core.utils.kaggle.data_repository import KaggleDataRepository


class KaggleDataRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.output_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/test"
		self.slug = "abrehamatlaw0/mock-notebook-0"
		self.repository = KaggleDataRepository(
			output_path=self.output_path
		)

	def test_download(self):
		self.repository.download()
		os.system(f"tree \"{self.output_path}\"")

	def test_checksum(self):
		download_path = os.path.join(self.output_path, self.slug.replace("/", "-"))
		self.repository.download("abrehamatlaw0/mock-notebook-0")
		checksum = self.repository.generate_checksum(download_path)
		print(f"Checksum {checksum}")
		self.repository.download(self.slug, checksum=checksum)

