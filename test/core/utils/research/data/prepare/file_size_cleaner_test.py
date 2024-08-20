import os
import random
import unittest

import numpy as np

from core.utils.research.data.prepare.file_size_cleaner import FileSizeCleaner


class FileSizeCleanerTest(unittest.TestCase):

	def setUp(self):
		self.files = self.__generate_files("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/dummy")

	def __generate_files(self, path):
		paths = []
		for i in range(10):
			arr = np.random.random((random.randint(5, 10), 5))
			save_path = os.path.join(path, f"{i}.npy")
			np.save(save_path, arr)
			paths.append(save_path)

		return paths

	def test_functionality(self):

		cleaner = FileSizeCleaner()
		cleaner.start(self.files)
