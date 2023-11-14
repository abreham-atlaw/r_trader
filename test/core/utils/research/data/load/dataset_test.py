import unittest

from core.utils.research.data.load.dataset import BaseDataset


class BaseDatasetTest(unittest.TestCase):

	def test_functionality(self):

		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)

		X, y = dataset[1500]
		self.assertEqual(X.shape[0], 1024)
		self.assertEqual(y.shape[0], 449)
