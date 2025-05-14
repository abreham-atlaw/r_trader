import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.utils.research.data.load.dataset import BaseDataset


class BaseDatasetTest(unittest.TestCase):

	def setUp(self):
		self.dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			num_files=3,
			load_weights=True,
			return_weights=False
		)

	def test_functionality(self):

		X, y, w = self.dataset[15]

		print(X, y, w)
		self.assertEqual(X.shape[0], 1148)
		self.assertEqual(y.shape[0], 432)
		self.assertIsNotNone(w)

	def test_shuffle(self):
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			return_weights=False
		)

		X_0, y_0 = dataset[54]

		dataset.shuffle()
		X_1, y_1 = dataset[54]

		dataset.shuffle()
		X_2, y_2 = dataset[54]

		for i, (X, y) in enumerate(zip([X_0, X_1, X_2], [y_0, y_1, y_2])):
			for j, (X_o, y_o) in enumerate(zip([X_0, X_1, X_2], [y_0, y_1, y_2])):
				if i == j:
					continue
				self.assertFalse(torch.all(X == X_o))
				self.assertFalse(torch.all(y == y_o))

	def test_dataloader(self):
		dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=6
		)

		self.dataset.shuffle()

		for X, y, w in dataloader:
			self.assertIsNotNone(X)



