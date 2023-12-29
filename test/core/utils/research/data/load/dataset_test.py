import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.utils.research.data.load.dataset import BaseDataset


class BaseDatasetTest(unittest.TestCase):

	def test_functionality(self):

		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
			num_files=3
		)

		X, y = dataset[1500]
		self.assertEqual(X.shape[0], 1024)
		self.assertEqual(y.shape[0], 449)

	def test_shuffle(self):
		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)

		X_0, y_0 = dataset[1500]

		dataset.shuffle()
		X_1, y_1 = dataset[1500]

		dataset.shuffle()
		X_2, y_2 = dataset[1500]

		for i, (X, y) in enumerate(zip([X_0, X_1, X_2], [y_0, y_1, y_2])):
			for j, (X_o, y_o) in enumerate(zip([X_0, X_1, X_2], [y_0, y_1, y_2])):
				if i == j:
					continue
				self.assertFalse(torch.all(X == X_o))
				self.assertFalse(torch.all(y == y_o))

	def test_dataloader(self):
		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)

		dataloader = DataLoader(
			dataset=dataset,
			batch_size=6
		)

		dataset.shuffle()

		for X, y in dataloader:
			self.assertIsNotNone(X)



