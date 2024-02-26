import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.model.model.cnn.model import CNN
from lib.utils.torch_utils.model_handler import ModelHandler


class CNNTest(unittest.TestCase):

	def test_functionality(self):

		# CHANNELS = [128, 128]
		# KERNEL_SIZES = [3 for _ in CHANNELS]
		# BLOCK_SIZE = 64
		# VOCAB_SIZE = 51
		# POOL_SIZES = [0 for _ in CHANNELS]
		# DROPOUT_RATE = 0
		# ACTIVATION = nn.ReLU()
		#
		# model = CNN(
		# 	num_classes=VOCAB_SIZE,
		# 	conv_channels=CHANNELS,
		# 	kernel_sizes=KERNEL_SIZES,
		# 	hidden_activation=ACTIVATION,
		# 	pool_sizes=POOL_SIZES,
		# 	dropout_rate=DROPOUT_RATE
		# )
		# model.load_state_dict(
		# 	torch.load("/home/abreham/Downloads/model(34).pth", map_location=torch.device('cpu'))
		# )
		model = ModelHandler.load("/home/abreham/Downloads/Compressed/bemnetatlaw-rtrader-cnn-0.zip")

		DTYPE = torch.float32
		NP_DTYPE = np.float32

		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared(64)/train/X/1703744358.894403.npy").astype(NP_DTYPE)[:]
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared(64)/train/y/1703744358.894403.npy").astype(NP_DTYPE)[:]

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		y_hat_classes = np.argmax(y_hat.detach().numpy(), axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertEqual(y_hat_classes.shape, (X.shape[0],))
