import unittest

import numpy as np

import os

from core.utils.research.data.prepare.duplicate_data_cleaner import DuplicateDataCleaner


class DuplicateDataCleanerTest(unittest.TestCase):

	def setUp(self):
		self.files = self.__generate_data("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/dummy", 10)

	def __generate_data(self, path: str, size: int):
		def generate_file(i, path):
			X = np.arange(i, i+10).reshape((5, 2))
			y = X[:, 0]

			paths = tuple([os.path.join(path, f'{i}-{t}.npy') for t in ['X', 'y']])
			for arr, s_path in zip((X, y), paths):
				np.save(s_path, arr)

			return paths

		files = [
			generate_file(i, path)
			for i in range(size)
		]

		return tuple([[p[i] for p in files] for i in range(2)])

	def test_functionality(self):
		cleaner = DuplicateDataCleaner()
		cleaner.start(self.files[0], self.files[1])
