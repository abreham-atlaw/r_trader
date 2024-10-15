import unittest

import numpy as np

from core.utils.research.data.prepare.duplicate_cleaner import DuplicateCleaner


class DuplicateCleanerTest(unittest.TestCase):

	def test_functionality(self):

		original_data = [np.arange(0, 10).reshape(5, 2)]
		dirty_data = np.array([
			[4, 5],
			[8, 9],
			[1, 2]
		])
		clean_data = np.array([
			[1, 2]
		])
		companion_data = np.array([3, 5, 7])
		clean_companion = np.array([7])
		cleaner = DuplicateCleaner(arrays=original_data)
		cleaned_data, cleaned_companion = cleaner.clean(dirty_data, companion_array=companion_data)

		dirty_data_2 = np.array([
			[1, 2],
			[0, 8]
		])
		clean_data_2 = np.array([
			[0, 8]
		])

		self.assertTrue(np.all(clean_data == cleaned_data))
		self.assertTrue(np.all(clean_companion == cleaned_companion))

		cleaned_data_2 = cleaner.clean(dirty_data_2)
		self.assertTrue(np.all(clean_data_2 == cleaned_data_2))

	def test_fresh_start(self):
		dirty_data = np.array([
			[4, 5],
			[8, 9],
			[1, 2]
		])
		clean_data = np.array([
			[4, 5],
			[8, 9],
			[1, 2]
		])
		companion_data = np.array([3, 5, 7])
		clean_companion = np.array([3, 5, 7])

		dirty_data_2 = np.array([
			[1, 2],
			[0, 8]
		])
		clean_data_2 = np.array([
			[0, 8]
		])

		cleaner = DuplicateCleaner()
		cleaned_data, cleaned_companion = cleaner.clean(dirty_data, companion_array=companion_data)

		self.assertTrue(np.all(clean_data == cleaned_data))
		self.assertTrue(np.all(clean_companion == cleaned_companion))

		cleaned_data_2 = cleaner.clean(dirty_data_2)
		self.assertTrue(np.all(clean_data_2 == cleaned_data_2))
