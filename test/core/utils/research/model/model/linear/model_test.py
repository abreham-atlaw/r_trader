import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.losses import WeightedMSELoss
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.wrapped import WrappedModel
from lib.utils.torch_utils.model_handler import ModelHandler
import matplotlib.pyplot as plt


class LinearTest(unittest.TestCase):

	def test_functionality(self):

		def softmax(x):
			exp_x = np.exp(x - np.max(x))
			softmax_x = exp_x / np.sum(exp_x)
			return softmax_x

		def scale(x):
			x = softmax(x)
			x = x / np.max(x)
			return x
		# LAYER_SIZES = [256, 256]
		# NORM = [True] + [False for _ in LAYER_SIZES]
		# BLOCK_SIZE = 1148
		# VOCAB_SIZE = 449
		# DROPOUT_RATE = 0
		# ACTIVATION = nn.ReLU()
		# INIT = None
		#
		# model = LinearModel(
		# 	block_size=BLOCK_SIZE,
		# 	vocab_size=VOCAB_SIZE,
		# 	dropout_rate=DROPOUT_RATE,
		# 	layer_sizes=LAYER_SIZES,
		# 	hidden_activation=ACTIVATION,
		# 	init_fn=INIT,
		# 	norm=NORM
		# )

		# ModelHandler.save(model, "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/bemnetatlaw-drmca-linear-0.zip")
		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-43-cum-0-it-2-tot.zip")
		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-42-cum-0-it-2-tot.zip")

		DTYPE = torch.float32
		NP_DTYPE = np.float32

		SIZE = 10

		X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/train/X/1727611458.762627.npy").astype(NP_DTYPE)[:SIZE]
		y = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/drl_export/train/y/1727611458.762627.npy").astype(NP_DTYPE)[:SIZE]

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X)).detach().numpy()

		softmaxed = np.array([softmax(y_hat[i, :-1]) for i in range(y_hat.shape[0])])
		scaled = np.array([scale(y_hat[i, :-1]) for i in range(y_hat.shape[0])])

		for i in range(y_hat.shape[0]):
			plt.figure()
			plt.plot(y[i, :-1])
			plt.plot(softmaxed[i])

			plt.figure()
			plt.plot(X[i, :-124])

		plt.show()

		y_hat_classes = np.argmax(y_hat, axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertEqual(y_hat_classes.shape, (X.shape[0],))
