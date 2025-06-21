import os
import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import math

from core import Config
from core.utils.research.data.prepare.swg.manipulate import ReciprocateSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline
from core.utils.research.data.prepare.swg.xswg import MomentumXSampleWeightGenerator
from core.utils.research.losses import ProximalMaskedLoss
from core.utils.research.model.layers import L1Norm
from core.utils.research.model.model.ensemble.stacked.msm.performance_grid_msm import PerformanceGridMSM
from core.utils.research.utils.ensemble.pg_msm.pg_msm_builder import PerformanceGridMSMBuilder
from lib.utils.torch_utils.model_handler import ModelHandler


class PerformanceGridMSMBuilderTest(unittest.TestCase):

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

		self.data_paths = [
			f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/{i}/train"
			for i in [4, 5]
		]

		self.tmp_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/tmp"

		self.generators = [
			MomentumXSampleWeightGenerator(),
			SampleWeightGeneratorPipeline(
				generators=[
					MomentumXSampleWeightGenerator(),
					ReciprocateSampleWeightManipulator(),
				]
			)
		]

		self.builder = PerformanceGridMSMBuilder(
			data_paths=self.data_paths,
			generators=self.generators,
			tmp_path=self.tmp_path,
			loss=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				p=1,
				weighted_sample=False
			)
		)

		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/5/train/X/1743180011.758197.npy").astype(np.float32))
		self.y = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/5/train/y/1743180011.758197.npy").astype(np.float32))
		self.MODEL_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/pg_msm-0.zip"

	def test_functionality(self):

		model = self.builder.build(
			models=self.models,
			activation=L1Norm()
		)
		model.eval()

		self.assertIsInstance(model, PerformanceGridMSM)
		self.assertIsInstance(model.activation, L1Norm)

		y = model(self.X)
		print(y)
		self.assertIsNotNone(y)

		ModelHandler.save(model, self.MODEL_PATH)
		loaded_model = ModelHandler.load(self.MODEL_PATH)
		loaded_model.eval()
		y_loaded = loaded_model(self.X)

		self.assertTrue(torch.all(y == y_loaded))
	
	def test_analyze_output(self):

		model = self.builder.build(
			models=self.models,
			activation=L1Norm()
		)
		model.eval()

		softmax = nn.Softmax(dim=-1)

		y = softmax(model(self.X))
		ys = [
			softmax(m(self.X))
			for m in self.models
		]

		performance_grid = np.load(os.path.join(self.tmp_path, "performance_grid.npy"))

		plots = len(ys) + 2
		cols = 3
		rows = math.ceil(plots/cols)

		samples = 3

		for j in range(samples):
			plt.figure()

			plt.subplot(rows, cols, 1)
			plt.imshow(performance_grid)
			plt.colorbar()

			for i, y_m in enumerate(ys + [y]):
				plt.subplot(rows, cols, i+2)

				for y_ in [self.y, y_m, y]:
					plt.plot(y_[j, :].detach().numpy())

		plt.show()
