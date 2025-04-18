import unittest

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.eval.mlpl_evaluator import MLPLEvaluator
from core.utils.research.eval.mlpl_evaluator.losses.unbatched_pml_loss import UnbatchedProximalMaskedLoss
from core.utils.research.losses import ProximalMaskedLoss
from lib.utils.torch_utils.model_handler import ModelHandler


class MLPLEvaluatorTest(unittest.TestCase):

	def setUp(self):

		self.models1 = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip"
			]
		]
		self.models2 = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-6-tot_1.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip"
			]
		]
		self.dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		self.dataloader = DataLoader(self.dataset, batch_size=1)
		self.dataloader_b4 = DataLoader(self.dataset, batch_size=4)
		self.evaluator = MLPLEvaluator(
			loss=UnbatchedProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				p=1,
				softmax=True,
			),
			# loss=nn.CrossEntropyLoss(),
			dataloader=self.dataloader,
			progress_interval=20,
			losses_export_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/losses_export"
		)
		self.evaluator_b4 = MLPLEvaluator(
			loss=UnbatchedProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				p=1,
				softmax=True,
			),
			dataloader=self.dataloader_b4,
			progress_interval=20,
			losses_export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/losses_export"
		)

	def test_evaluate(self):

		loss = self.evaluator.evaluate(self.models2)
		self.assertIsNotNone(loss)

	def test_evaluate_relative_value(self):

		losses = [
			self.evaluator.evaluate(models)
			for models in [self.models1, self.models2]
		]
		print(losses)
		self.assertGreater(1, 0)
		self.assertGreater(losses[0], losses[1])

	def test_batchsize_equivalence(self):

		losses = [
			evaluator.evaluate(self.models2)
			for evaluator in [
				self.evaluator,
				self.evaluator_b4
			]
		]

		self.assertTrue(losses[0], losses[1])