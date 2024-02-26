import unittest

import torch

from core.utils.research.losses import WeightedMSELoss


class WeightedMSELossTest(unittest.TestCase):


	def test_functionality(self):
		y = torch.asarray([[0, 1, 0, 0]])
		y_hat = torch.asarray([[0.1, 0.1, 0.7, 0.1]])

		loss = WeightedMSELoss(4)
		e = loss(y, y_hat)
		self.assertIsNotNone(e)