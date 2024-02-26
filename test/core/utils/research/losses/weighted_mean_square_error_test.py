import unittest

import torch

from core.utils.research.losses import WeightedMSELoss


class WeightedMSELossTest(unittest.TestCase):


	def test_functionality(self):
		y = torch.asarray([[0, 1, 0, 0], [0, 1, 0, 0]])
		y_hat = torch.asarray([[0, 0.9, 0.1, 0], [0, 1, 0, 0]])

		loss = WeightedMSELoss(4, a=0.001)
		e = loss(y_hat, y)
		self.assertIsNotNone(e)
