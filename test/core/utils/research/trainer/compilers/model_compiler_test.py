import unittest

import numpy as np
import torch

from core.utils.research.training.compilers import ModelCompiler
from core.utils.research.training.compilers.simplify_compiler import SimplifyCompiler
from core.utils.research.training.compilers.ts_compiler import TorchScriptCompiler
from lib.utils.torch_utils.model_handler import ModelHandler


class ModelCompilerTest(unittest.TestCase):

	def setUp(self):
		NP_DTYPE = np.float32
		self.model = SimplifyCompiler().compile(
			ModelHandler.load(
				# "/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-85-cum-0-it-4-tot.zip"
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-151-cum-0-it-4-tot.zip"
			)
		)
		self.model.eval()
		# self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-151-cum-0-it-4-tot.zip")
		self.X = torch.from_numpy(np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/X/1727815242.844215.npy").astype(
			NP_DTYPE))
		self.y = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/2/test/y/1727815242.844215.npy").astype(
			NP_DTYPE)
		self.compiler = TorchScriptCompiler()

	def test_compile(self):
		compiled = self.compiler.compile(self.model)
		compiled.eval()
		self.assertIsNotNone(compiled)

		outs = [
			model(self.X)
			for model in [compiled, self.model]
		]

		self.assertTrue(
			torch.all(
				torch.eq(outs[0], outs[1])
			)
		)
