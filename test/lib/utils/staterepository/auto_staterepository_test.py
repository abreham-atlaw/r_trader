import unittest

from lib.utils.staterepository import AutoStateRepository


class AutoStateRepositoryTest(unittest.TestCase):

	def setUp(self):
		self.repository = AutoStateRepository(memory_size=5)
