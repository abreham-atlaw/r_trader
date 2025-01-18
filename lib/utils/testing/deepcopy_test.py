from abc import abstractmethod, ABC

import unittest
from copy import deepcopy

from .method import Method
from ..logger import Logger


class DeepcopyTest(unittest.TestCase, ABC):

	@abstractmethod
	def _get_instance(self) -> object:
		pass

	@abstractmethod
	def _get_write_method(self) -> Method:
		pass

	def setUp(self):
		super().setUp()
		self.instance = self._get_instance()
		self.write_method = self._get_write_method()

	def __test_deepcopy(self):
		Logger.info(f"Checking deepcopy of '{self.instance.__class__}'...")
		copy = deepcopy(self.instance)
		self.assertEqual(self.instance, copy)
		self.write_method.call(copy)
		self.assertNotEqual(self.instance, copy)
		Logger.success(f"deepcopy of '{self.instance.__class__}' passed!")

	def __test_hash(self):
		Logger.info(f"Checking hash of '{self.instance.__class__}'...")
		copy = deepcopy(self.instance)
		self.assertEqual(hash(self.instance), hash(copy))
		self.write_method.call(copy)
		Logger.success(f"hash of '{self.instance.__class__}' passed!")
		self.assertNotEqual(hash(self.instance), hash(copy))

	def start(self):
		self.__test_deepcopy()
		self.__test_hash()
