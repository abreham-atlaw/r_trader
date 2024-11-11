import unittest

import numpy as np
import torch

from core.utils.research.training.compilers.simplify_compiler import SimplifyCompiler
from lib.utils.torch_utils.model_handler import ModelHandler


class SimplifyCompilerTest(unittest.TestCase):

	def setUp(self):
		self.model = ModelHandler.load(
			"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-85-cum-0-it-4-tot.zip"
		)
		self.model.eval()
		self.compiler = SimplifyCompiler()
		NP_DTYPE = np.float32
		self.X = torch.from_numpy(np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/X/1727815242.844215.npy").astype(
			NP_DTYPE
		))

	def test_compile(self):
		simplified = self.compiler.compile(self.model)

		simplified.eval()

		outs = [
			model(self.X)
			for model in [simplified, self.model]
		]

		self.assertTrue(
			torch.all(
				torch.eq(outs[0], outs[1])
			)
		)
