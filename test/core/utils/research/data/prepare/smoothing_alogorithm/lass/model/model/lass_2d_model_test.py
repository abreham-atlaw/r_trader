import unittest

import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.models import Lass2DModel
from lib.utils.torch_utils.model_handler import ModelHandler


class Lass2DModelTest(unittest.TestCase):

	def setUp(self):
		self.model = Lass2DModel(
			ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-lass-training-cnn-0-it-3-tot.zip")
		)

	def test_functionality(self):
		X = torch.arange(0, self.model.input_size[-1]*2).reshape(2, -1).float()
		y = self.model(X)
		self.assertIsNotNone(y)
