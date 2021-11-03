from typing import *

import unittest
import cattr, attr

from lib.network.rest_interface import Serializer


@attr.define
class Model1:

	attrib0: str = attr.ib()
	attrib1: int = attr.ib()


class SerializerTest(unittest.TestCase):

	JSON = '{"attrib0": "hello", "attrib1": 1}'
	INSTANCE = Model1(attrib0="hello", attrib1=1)

	def setUp(self) -> None:
		self.model1_serializer = Serializer(Model1)

	def test_serialize(self):
		self.assertEqual(
			self.model1_serializer.serialize(SerializerTest.INSTANCE),
			SerializerTest.JSON
		)
	
	def test_deserialize(self):
		self.assertEqual(
			self.model1_serializer.deserialize(SerializerTest.JSON),
			SerializerTest.INSTANCE
		)


if __name__ == "__main__":
	unittest.main()