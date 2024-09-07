import unittest

import os

import numpy as np

from core.utils.research.data.prepare.files_merger import FilesMerger


class FilesMergerTest(unittest.TestCase):

	def setUp(self):
		self.path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/test/files_merger"
		all_paths = self.x_path, self.y_path, self.merged_path, self.x_split, self.y_split = [os.path.join(self.path, container) for container in ["x", "y", "merged", "x_split", "y_split"]]

		for path in all_paths:
			os.system(f"rm -rf {path}")
			os.mkdir(path)

		for i in range(10):
			X, y = np.random.random((5, 8)), np.random.random((5, 2))
			for arr, path in zip((X, y), [self.x_path, self.y_path]):
				np.save(os.path.join(path, f"{i}.npy"), arr)

	def test_merge(self):

		merger = FilesMerger()
		merger.merge(self.x_path, self.y_path, self.merged_path)

		self.assertEqual(os.listdir(self.merged_path), os.listdir(self.x_path))

	def test_split(self):

		merger = FilesMerger()
		merger.merge(self.x_path, self.y_path, self.merged_path)

		merger.split(self.x_split, self.y_split, self.merged_path)

		for filename in os.listdir(self.x_split):
			for split, original in zip([self.x_split, self.y_split], [self.x_path, self.y_path]):
				self.assertTrue(
					np.all(
						np.load(os.path.join(split, filename)) == np.load(os.path.join(original, filename))
					)
				)
