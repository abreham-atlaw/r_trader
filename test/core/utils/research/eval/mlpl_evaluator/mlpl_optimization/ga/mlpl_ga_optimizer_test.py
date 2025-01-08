import unittest

from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.eval.mlpl_evaluator import MLPLEvaluator
from core.utils.research.eval.mlpl_evaluator.mlpl_optimization.ga import MLPLGAOptimizer
from core.utils.research.losses import ProximalMaskedLoss


class MLPLGAOptimizerTest(unittest.TestCase):

	def setUp(self):
		MODELS = [
			f"cnn-{i}"
			for i in [148, 168, 222, 264]
		]
		MODEL_NAMES = [
			f"abrehamalemu-rtrader-training-exp-0-{model}-cum-0-it-4-tot.zip"
			for model in MODELS
		]

		self.dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		self.dataloader = DataLoader(self.dataset, batch_size=1)
		self.evaluator = MLPLEvaluator(
			loss=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				p=1,
				softmax=True,
			),
			# loss=nn.CrossEntropyLoss(),
			dataloader=self.dataloader
		)

		self.optimizer = MLPLGAOptimizer(
			models=MODEL_NAMES,
			population_size=10,
			evaluator=self.evaluator
		)

	def test_optimize(self):
		self.optimizer.start(epochs=5)
