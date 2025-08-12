import unittest

import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers import SmoothedChannelDropout


class SmoothedChannelDropoutTest(unittest.TestCase):

	def setUp(self):
		self.layer = SmoothedChannelDropout(0.9, 0.9)

	def test_functionality(self):
		x = torch.rand(5, 2, 10)
		y = self.layer(x)
		self.assertEqual(x.shape, y.shape)
