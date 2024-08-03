import unittest

import torch

from core.utils.research.losses import WeightedMSELoss


class WeightedMSELossTest(unittest.TestCase):


	def test_functionality(self):

		true_class = 0
		num_classes = 10
		y = torch.asarray([[0 if i != true_class else 1 for i in range(num_classes)]])
		y_hats = [
			torch.asarray([[0 if i != j else 1 for i in range(num_classes)]])
			for j in range(num_classes)
		]

		loss = WeightedMSELoss(num_classes, a=1, softmax=False)
		losses = [loss(y_hat, y) for y_hat in y_hats]
		self.assertIsNotNone(losses)

		self.assertGreater(losses[0], losses[1])
