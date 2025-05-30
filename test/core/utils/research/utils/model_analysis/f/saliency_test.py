import unittest

import numpy as np
import torch

from core.utils.research.utils.model_analysis.f import get_layer_io_saliency
from lib.utils.torch_utils.model_handler import ModelHandler


class SaliencyTest(unittest.TestCase):

	def setUp(self):
		self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-37-cum-0-it-35-tot.zip")
		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/6/train/X/1740913843.59131.npy").astype(np.float32))

	def test_layer_io_saliency(self):
		s = [
			get_layer_io_saliency(self.model, self.X, layer)
			for layer in self.model.layers
		]
		self.assertIsNotNone(s)
