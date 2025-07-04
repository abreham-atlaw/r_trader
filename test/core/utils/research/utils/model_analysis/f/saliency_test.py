import unittest

import numpy as np
import torch

from core.utils.research.utils.model_analysis.f import get_layer_io_saliency, get_layer_saliency
from lib.utils.torch_utils.model_handler import ModelHandler


class SaliencyTest(unittest.TestCase):

	def setUp(self):
		self.model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-4-it-42-tot.zip")
		self.X = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/7/train/X/1751195327.143124.npy").astype(np.float32))

	def test_layer_saliency(self):
		s = get_layer_saliency(self.model, self.X, self.model.bridge_block.transformer_block.decoder_block.self_attention_layer)
		self.assertIsNotNone(s)

	def test_layer_io_saliency(self):
		s = [
			get_layer_io_saliency(self.model, self.X, layer)
			for layer in self.model.layers
		]
		self.assertIsNotNone(s)
