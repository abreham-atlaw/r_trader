import json
import math
import os
import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from core import Config
from core.utils.research.model.model.utils import HorizonModel
from lib.utils.torch_utils.model_handler import ModelHandler


class HorizonModelTest(unittest.TestCase):

	def setUp(self):

		with open(os.path.join(Config.BASE_DIR, "res/bounds/05.json"), "r") as f:
			bounds = json.load(f)
		self.og_model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-1-it-42-tot.zip")
		self.og_model.eval()
		self.model = HorizonModel(
			model=ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-1-it-42-tot.zip"),
			bounds=bounds,
			h=0.2
		)
		self.model.eval()
		self.X = torch.from_numpy(np.load(
			os.path.join(Config.BASE_DIR, "temp/Data/prepared/7/train/X/1751195327.143124.npy")
		).astype(np.float32))

	def test_process_sample(self):
		x = self.model.process_sample(self.X.clone())
		self.assertIsNotNone(x)
		self.assertTrue(torch.all(x[:, :-125] == self.X[:, :-125]))
		self.assertTrue(torch.all(x[:, -125] != self.X[:, -125]))

		samples = 6
		plt.figure()
		for j in range(samples):
			plt.subplot(math.ceil(samples / 3), 3, j + 1)
			for i, arr in enumerate([self.X, x]):
				plt.plot(arr[j, :-124][-100:].detach().numpy())
		plt.show()

	def test_functionality(self):

		softmax = nn.Softmax(dim=-1)

		y = self.model(self.X)
		y_hat = self.og_model(self.X)

		samples = 6

		y, y_hat = [softmax(arr).detach().numpy() for arr in [y, y_hat]]

		plt.figure()
		for j in range(samples):
			plt.subplot(math.ceil(samples/3), 3,  j + 1)
			for i, arr in enumerate([y, y_hat]):
				plt.plot(arr[j])
		plt.show()
