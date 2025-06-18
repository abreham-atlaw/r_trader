import unittest

import numpy as np
import torch

from core import Config
from core.utils.research.eval.mlpl_evaluator.losses.unbatched_pml_loss import UnbatchedProximalMaskedLoss
from core.utils.research.losses import ProximalMaskedLoss
from core.utils.research.utils.unbatched_model_evaluator import UnbatchedModelEvaluator
from lib.utils.torch_utils.model_handler import ModelHandler


class UnbatchedModelEvaluatorTest(unittest.TestCase):

	def setUp(self):
		self.evaluator = UnbatchedModelEvaluator(
			cls_loss_fn=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				p=1,
				collapsed=False
			),
			data_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
		)
		self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-45-cum-0-it-29-tot.zip")

	def test_functionality(self):

		losses = self.evaluator.evaluate(self.model)
		print(losses)

		self.assertIsInstance(losses, np.ndarray)
		self.assertEqual(losses.shape[0], 128)
