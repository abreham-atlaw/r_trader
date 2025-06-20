import unittest

import numpy as np
from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.losses import ProximalMaskedLoss
from core.utils.research.utils.ensemble.pg_msm.performance_grid_evaluator import PerformanceGridEvaluator
from lib.utils.torch_utils.model_handler import ModelHandler


class PerformanceGridEvaluatorTest(unittest.TestCase):

	def setUp(self):
		self.dataloaders = [
			DataLoader(
				dataset=BaseDataset(
					root_dirs=[
						f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/{i}/train"
					],
					load_weights=True
				)
			)
			for i in [4, 5]
		]

		self.evaluator = PerformanceGridEvaluator(
			dataloaders=self.dataloaders,
			loss=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				p=1,
				weighted_sample=True
			)
		)

		self.models = [
			ModelHandler.load(model_path)
			for model_path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-69-cum-0-it-27-sw12-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-40-cum-0-it-27-sw12-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-35-cum-0-it-27-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-36-cum-0-it-35-tot.zip"
			]
		]
		self.export_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/performance_grid.npy"

	def test_functionality(self):

		performance = self.evaluator.evaluate(
			self.models,
			export_path=self.export_path
		)
		print(performance)

		loaded_performance = np.load(self.export_path)
		self.assertTrue(np.array_equal(performance, loaded_performance))
		self.assertTrue(performance.shape == (len(self.models), len(self.dataloaders)))





