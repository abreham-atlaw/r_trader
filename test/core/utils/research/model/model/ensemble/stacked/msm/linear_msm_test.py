import unittest

import numpy as np
import torch

from core.utils.research.model.model.ensemble.stacked import LinearMSM
from core.utils.research.model.model.linear.model import LinearModel
from lib.utils.torch_utils.model_handler import ModelHandler


class LinearMSMTest(unittest.TestCase):

	def setUp(self):
		self.MODEL_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/ensemble_stacked_linear_musm.zip"
		self.MODELS = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/results_4/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-4-tot_1.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_3/abrehamalemu-rtrader-training-exp-0-cnn-173-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip"
			]
		]
		self.model = LinearMSM(
			models=self.MODELS,
			ff=LinearModel(
				layer_sizes=[512, 256]
			)
		)
		self.X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1724671615.45445.npy").astype(
			np.float32)

	def test_call(self):
		self.model.eval()
		with torch.no_grad():
			y_hat = self.model(torch.from_numpy(self.X))

		self.assertIsNotNone(y_hat)
		self.assertEqual(y_hat.shape[0], self.X.shape[0])

		ModelHandler.save(self.model, self.MODEL_PATH)

	def test_load(self):
		model = ModelHandler.load(self.MODEL_PATH)
		model.eval()
		with torch.no_grad():
			y_hat = model(torch.from_numpy(self.X))

		self.assertEqual(len(self.model.models), len(model.models))

		self.assertIsNotNone(model)
