import unittest

import torch
import numpy as np

from core.utils.research.model.model.utils import TransitionOnlyModel
from lib.utils.torch_utils.model_handler import ModelHandler


class TransitionOnlyModelTest(unittest.TestCase):

	def setUp(self):
		self.bare_model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-94-cum-0-it-4-tot.zip")
		self.bare_model.eval()
		self.model = TransitionOnlyModel(
			model=self.bare_model,
			extra_len=124
		)
		self.X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/1/test/X/1728122227.228115.npy").astype(np.float32)

	def test_call(self):
		y = self.model(torch.from_numpy(self.X)).detach().numpy()

		self.assertTrue(np.all(y[:, -1] == 0))

	def test_comparison(self):
		raw_y, y = [
			model(torch.from_numpy(self.X)).detach().numpy()
			for model in [self.bare_model, self.model]
		]
		difference = raw_y - y

		self.assertEqual(raw_y.shape, y.shape)

