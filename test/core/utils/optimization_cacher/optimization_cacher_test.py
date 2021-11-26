from typing import *

import unittest

from core.utils.optimization_cacher.OptimizationCacher import OptimizationCacher


class OptimizationCacherTest(unittest.TestCase):

	CONFIG_DICT = {
		'seq_len': 4,
		'hidden_layers': [4],
		'loss': 'binary_crossentropy',
		'optimizer': 'adam',
		'hidden_activation': 'relu',
		'delta': True,
		'percentage': True,
		'average_window': 0
	}
	VALUE = 0.6918424665927887
	USER = "test_user"

	def setUp(self):
		self.cacher = OptimizationCacher()

	def test_cache(self):
		self.cacher.cache(
			OptimizationCacherTest.CONFIG_DICT,
			OptimizationCacherTest.VALUE
		)
		value = self.cacher.get_value(
			OptimizationCacherTest.CONFIG_DICT
		)

		self.assertIsNotNone(value)

		self.assertAlmostEqual(
			OptimizationCacherTest.VALUE,
			value
		)

	def test_lock(self):
		self.cacher.lock(
			OptimizationCacherTest.CONFIG_DICT,
			OptimizationCacherTest.USER
		)

		self.assertTrue(
			self.cacher.is_locked(
				OptimizationCacherTest.CONFIG_DICT
			)
		)

		self.cacher.cache(
			OptimizationCacherTest.CONFIG_DICT,
			OptimizationCacherTest.VALUE
		)

		self.assertFalse(
			self.cacher.is_locked(
				OptimizationCacherTest.CONFIG_DICT
			)
		)
