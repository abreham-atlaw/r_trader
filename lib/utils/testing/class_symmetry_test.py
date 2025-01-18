import typing
from abc import ABC, abstractmethod

import unittest
from dataclasses import dataclass, field

import numpy as np

from lib.utils.logger import Logger
from .method import Method


class ClassSymmetryTest(unittest.TestCase, ABC):

	LOOSE_DECIMALS = 6

	@abstractmethod
	def _get_classes(self) -> typing.Tuple[type, type]:
		pass

	@abstractmethod
	def _get_constructor_arguments(self) -> typing.Tuple[typing.Tuple, typing.Dict]:
		pass
	
	@abstractmethod
	def _get_write_methods(self) -> typing.List[Method]:
		pass

	@abstractmethod
	def _get_read_methods(self) -> typing.List[Method]:
		pass

	@abstractmethod
	def _get_fields(self) -> typing.List[str]:
		pass

	def _create_instances(self):
		classes = self._get_classes()
		args, kwargs = self._get_constructor_arguments()
		return self._initialize_objects(classes, args, kwargs)

	def _initialize_objects(self, classes: typing.Tuple[type, type], args, kwargs) -> typing.Tuple[object, object]:
		return tuple([
			cls(*args, **kwargs)
			for cls in classes
		])

	def __compare_values(self, value1, value2, loose: bool, matrix: bool):
		loose_fn, non_loose_fn = self.assertAlmostEqual, self.assertEqual
		loose_decimal_key = "places" 
		if matrix:
			loose_fn, non_loose_fn = np.testing.assert_array_almost_equal, np.testing.assert_array_equal
			loose_decimal_key = "decimal"

		if loose:
			loose_fn(value1, value2, **{loose_decimal_key: self.LOOSE_DECIMALS})
		else:
			non_loose_fn(value1, value2)

	def __call_method(self, instance: object, method: Method) -> object:
		return getattr(instance, method.name)(*method.args, **method.kwargs)

	def __test_method_return_symmetry(self, method: Method):
		Logger.info(f"Checking return symmetry of '{method.name}'...")
		results = [
			self.__call_method(instance, method) 
			for instance in self.objects
		]
		self.__compare_values(results[0], results[1], method.loose, method.matrix)
		Logger.info(f"Method '{method.name}' tests passed!")

	def __test_field_symmetry(self, field: str):
		Logger.info(f"Checking field symmetry of '{field}'...")
		values = [getattr(instance, field) for instance in self.objects]
		self.assertEqual(values[0], values[1])
		Logger.info(f"Field '{field}' tests passed!\n\n\n\n")

	def __check_symmetry(self, method: typing.Optional[Method] = None):
		method_name = method.name if method is not None else "None"
		Logger.info(f"Checking symmetry of '{method_name}'...")

		if method is not None:
			self.__test_method_return_symmetry(method)

		for read_method in self.read_methods:
			self.__test_method_return_symmetry(read_method)

		for field in self.fields:
			self.__test_field_symmetry(field)
		Logger.info(f"Method '{method_name}' tests passed!\n\n\n\n")

	def setUp(self):
		super().setUp()
		self.objects = self._create_instances()
		self.read_methods = self._get_read_methods()
		self.write_methods = self._get_write_methods()
		self.fields = self._get_fields()

	def start(self):
		Logger.info(f"Checking symmetry of '{self.__class__.__name__}'...")
		self.__check_symmetry()
		for write_method in self.write_methods:
			self.__check_symmetry(write_method)
