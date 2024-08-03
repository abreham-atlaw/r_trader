import unittest

from core import Config
from core.di import ServiceProvider


class FileStorageTest(unittest.TestCase):

	def setUp(self):
		self.fs = ServiceProvider.provide_file_storage(path="/")

	def test_listdir(self):
		files = self.fs.listdir("/")
		self.assertTrue(len(files) > 0)

