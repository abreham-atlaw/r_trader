import math
import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator, \
	ReciprocateSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline
from core.utils.research.data.prepare.swg.xswg import MomentumXSampleWeightGenerator
from core.utils.research.model.layers import L1Norm
from core.utils.research.model.model.ensemble.stacked.msm.performance_grid_msm import PerformanceGridMSM
from lib.utils.torch_utils.model_handler import ModelHandler


class PerformanceGridMSMTest(unittest.TestCase):

	def setUp(self):
		self.models = [
			ModelHandler.load(model_path)
			for model_path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-69-cum-0-it-27-sw12-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-40-cum-0-it-27-sw12-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-35-cum-0-it-27-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-36-cum-0-it-35-tot.zip"
			]
		]
		self.performance_grid = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/performance_grid.npy").astype(np.float32)
		self.weight_generators = [
			SampleWeightGeneratorPipeline(
				generators=[
					MomentumXSampleWeightGenerator(),
					StandardizeSampleWeightManipulator(
						target_std=0.3,
						target_mean=1.0,
						current_std=0.76,
						current_mean=1.01
					)
				]
			),
			SampleWeightGeneratorPipeline(
				generators=[
					MomentumXSampleWeightGenerator(),
					ReciprocateSampleWeightManipulator(),
					StandardizeSampleWeightManipulator(
						target_std=0.3,
						target_mean=1.0,
						current_std=0.76,
						current_mean=1.01
					)
				]
			)
		]

		self.model = PerformanceGridMSM(
			generators=self.weight_generators,
			performance_grid=self.performance_grid,
			models=self.models,
			activation=L1Norm(),
			pre_weight_softmax=True
		)
		self.model.eval()

		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/5/train/X/1743180011.758194.npy").astype(np.float32))
		self.y = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/5/train/y/1743180011.758194.npy").astype(np.float32))
		self.MODEL_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/pg_msm-0.zip"

	def test_functionality(self):

		y = self.model(self.X)

		self.assertIsNotNone(y)
		print(y)
		ModelHandler.save(self.model, self.MODEL_PATH)
		model = ModelHandler.load(self.MODEL_PATH)
		model.eval()
		y_loaded = model(self.X)

		self.assertTrue(torch.all(y == y_loaded))

	def test_analyze_output(self):

		softmax = nn.Softmax(dim=-1)

		y = softmax(self.model(self.X))
		ys = [
			softmax(m(self.X))
			for m in self.models
		]

		plots = len(ys) + 2
		cols = 3
		rows = math.ceil(plots/cols)

		samples = 3

		for j in range(samples):
			plt.figure()

			plt.subplot(rows, cols, 1)
			plt.imshow(self.performance_grid)
			plt.colorbar()

			for i, y_m in enumerate(ys + [y]):
				plt.subplot(rows, cols, i+2)

				for y_ in [self.y, y_m, y]:
					plt.plot(y_[j, :].detach().numpy())

		plt.show()
