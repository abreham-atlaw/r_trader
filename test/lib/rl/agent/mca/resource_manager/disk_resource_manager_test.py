import unittest
import uuid

from lib.rl.agent.mca.resource_manager import DiskResourceManager
from lib.utils.logger import Logger


class DiskResourceManagerTest(unittest.TestCase):

	def setUp(self):
		self.manager = DiskResourceManager(min_remaining_space=0.99)

	def test_has_resource(self):
		resource = self.manager.init_resource()
		self.assertTrue(self.manager.has_resource(resource))

	@staticmethod
	def __create_dummy_file(size):
		filename = f"{uuid.uuid4().hex}"
		with open(filename, "wb") as f:
			f.write(b'\0'*size*1024*1024)
		Logger.info(f"Created dummy file, {filename} of {size} MB")

	def test_no_resource(self):
		resource = self.manager.init_resource()
		self.__create_dummy_file(3*1024)
		self.assertFalse(self.manager.has_resource(resource))
