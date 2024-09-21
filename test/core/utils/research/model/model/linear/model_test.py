import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.losses import WeightedMSELoss
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.wrapped import WrappedModel
from lib.utils.torch_utils.model_handler import ModelHandler



class LinearTest(unittest.TestCase):

	def test_functionality(self):

		LAYER_SIZES = [256, 256]
		NORM = [True] + [False for _ in LAYER_SIZES]
		BLOCK_SIZE = 1148
		VOCAB_SIZE = 449
		DROPOUT_RATE = 0
		ACTIVATION = nn.ReLU()
		INIT = None

		model = LinearModel(
			dropout_rate=DROPOUT_RATE,
			layer_sizes=LAYER_SIZES,
			hidden_activation=ACTIVATION,
			init_fn=INIT,
			norm=NORM
		)

		# model = ModelHandler.load("/home/abreham/Downloads/Compressed/bemnetatlaw-rtrader-linear-wl-0.zip")
		ModelHandler.save(model, "/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-24-cum-0-it-2-tot.zip")

		loss_fn = WeightedMSELoss(size=449, a=0.01, softmax=True)
		DTYPE = torch.float32
		NP_DTYPE = np.float32

		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1704085283.024914.npy").astype(NP_DTYPE)[:]
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/y/1704085283.024914.npy").astype(NP_DTYPE)[:]

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		y_hat_classes = np.argmax(y_hat.detach().numpy(), axis=1)
		y_classes = np.argmax(y, axis=1)

		loss = loss_fn(y_hat, torch.from_numpy(y))
		self.assertEqual(y_hat_classes.shape, (X.shape[0],))

	def test_load_and_predict(self):

		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-linear-33-cum-0-it-2-tot.zip")

		NP_DTYPE = np.float32
		X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1724671615.45445.npy").astype(NP_DTYPE)
		y = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/y/1724671615.45445.npy").astype(NP_DTYPE)

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		self.assertEquals(y.shape, y_hat.shape)
