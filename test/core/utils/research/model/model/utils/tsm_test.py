import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from core.utils.research.model.model.utils import TemperatureScalingModel
from lib.utils.torch_utils.model_handler import ModelHandler


class TemperatureScalingModelTest(unittest.TestCase):

	def setUp(self):
		self.raw_model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/results_7/abrehamalemu-rtrader-training-exp-0-linear-102-cum-0-it-4-tot_1.zip")
		self.scaled_models = [
			TemperatureScalingModel(
				model=self.raw_model,
				temperature=i/10
			)
			for i in range(1, 11)
		]
		self.X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/1/test/X/1728122227.228115.npy").astype(np.float32)

	def test_functionality(self):

		outs = [
			model(torch.from_numpy(self.X)).detach().numpy()
			for model in self.scaled_models
		]

		for i in range(len(self.scaled_models)):
			plt.figure()
			plt.plot(outs[i][0])
			plt.title(f"Temperature = {i/10}")
		plt.show()

		self.assertIsNotNone(outs)

