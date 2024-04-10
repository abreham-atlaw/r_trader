import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.model.model.cnn.model import CNN
from lib.utils.torch_utils.model_handler import ModelHandler


class CNNTest(unittest.TestCase):

	def test_dummy(self):
		CHANNELS = [4, 4]
		SEQ_LEN = 24
		EXTRA_LEN = 3
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 10
		POOL_SIZES = [0 for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = nn.LeakyReLU()

		model = CNN(
			extra_len=EXTRA_LEN,
			num_classes=VOCAB_SIZE + 1,
			conv_channels=CHANNELS,
			kernel_sizes=KERNEL_SIZES,
			hidden_activation=ACTIVATION,
			pool_sizes=POOL_SIZES,
			dropout_rate=DROPOUT_RATE
		)

		X = torch.from_numpy(np.concatenate(
			(
				np.random.random((16, SEQ_LEN)).astype(np.float32),
				np.zeros((16, EXTRA_LEN)).astype(np.float32)
			),
			axis=1
		))

		y = model(X)



	def test_functionality(self):
		CHANNELS = [128, 128]
		EXTRA_LEN = 4
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 431
		POOL_SIZES = [0 for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = nn.LeakyReLU()

		model = CNN(
			extra_len=EXTRA_LEN,
			num_classes=VOCAB_SIZE + 1,
			conv_channels=CHANNELS,
			kernel_sizes=KERNEL_SIZES,
			hidden_activation=ACTIVATION,
			pool_sizes=POOL_SIZES,
			dropout_rate=DROPOUT_RATE
		)
		# model.load_state_dict(
		# 	torch.load("/home/abreham/Downloads/model(34).pth", map_location=torch.device('cpu'))
		# )
		# model = ModelHandler.load("/home/abreham/Downloads/Compressed/bemnetatlaw-rtrader-cnn-0.zip")
		ModelHandler.save(model, "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip")

		# DTYPE = torch.float32
		NP_DTYPE = np.float32
		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer/out/test/X/1712733579.455299.npy").astype(NP_DTYPE)
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer/out/test/y/1712733579.455299.npy").astype(NP_DTYPE)
		#
		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		self.assertEquals(y.shape, y_hat.shape)
