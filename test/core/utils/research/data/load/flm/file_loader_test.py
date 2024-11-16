import unittest

from core.utils.research.data.load.flm.file_loader import FileLoader


class FileLoaderTest(unittest.TestCase):

	def setUp(self):
		self.root_dirs = ["/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/dummy"]

		self.loader = FileLoader(
			root_dirs=self.root_dirs,
			pool_size=2
		)

	def test_functionality(self):

		for i in range(len(self.loader)):
			self.loader.load(i)
			data = self.loader[i]

			self.assertIsNotNone(data)

