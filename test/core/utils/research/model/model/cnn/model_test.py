import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from lib.utils.torch_utils.model_handler import ModelHandler


class CNNTest(unittest.TestCase):

	def test_dummy(self):
		# CHANNELS = [1024, 2048]
		# EXTRA_LEN = 4
		# KERNEL_SIZES = [3 for _ in CHANNELS]
		# VOCAB_SIZE = 431
		# POOL_SIZES = [0 for _ in CHANNELS]
		# DROPOUT_RATE = 0
		# ACTIVATION = nn.LeakyReLU()
		# INIT = None
		# BLOCK_SIZE = 1028
		# SEQ_LEN = BLOCK_SIZE - EXTRA_LEN
		#
		# FF_LINEAR_BLOCK_SIZE = 256
		# FF_LINEAR_OUTPUT_SIZE = 256
		# FF_LINEAR_LAYERS = [1024, 1024]
		# FF_LINEAR_ACTIVATION = nn.ReLU()
		# FF_LINEAR_INIT = None
		# FF_LINEAR_NORM = [True] + [False for _ in FF_LINEAR_LAYERS]

		# model = CNN(
		# 	extra_len=EXTRA_LEN,
		# 	num_classes=VOCAB_SIZE + 1,
		# 	conv_channels=CHANNELS,
		# 	kernel_sizes=KERNEL_SIZES,
		# 	hidden_activation=ACTIVATION,
		# 	pool_sizes=POOL_SIZES,
		# 	dropout_rate=DROPOUT_RATE,
		# 	ff_linear= LinearModel(
		# 		block_size=FF_LINEAR_BLOCK_SIZE,
		# 		vocab_size=FF_LINEAR_OUTPUT_SIZE,
		# 		dropout_rate=DROPOUT_RATE,
		# 		layer_sizes=FF_LINEAR_LAYERS,
		# 		hidden_activation=FF_LINEAR_ACTIVATION,
		# 		init_fn=FF_LINEAR_INIT,
		# 		norm=FF_LINEAR_NORM
		# 	)
		# )

		model = ModelHandler.load("/home/abreham/Downloads/Compressed/bemnetatlaw-drmca-cnn-2.zip")

		X = torch.from_numpy(np.concatenate(
			(
				np.random.random((16, SEQ_LEN)).astype(np.float32),
				np.zeros((16, EXTRA_LEN)).astype(np.float32)
			),
			axis=1
		))

		y = model(X)

		ModelHandler.save(model, "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/model.zip")

	def test_functionality(self):
		# CHANNELS = [128 for i in range(5)]
		# EXTRA_LEN = 4
		# KERNEL_SIZES = [3 for _ in CHANNELS]
		# VOCAB_SIZE = 431
		# POOL_SIZES = [3 for _ in CHANNELS]
		# DROPOUT_RATE = 0
		# ACTIVATION = nn.LeakyReLU()
		# INIT = None
		# BLOCK_SIZE = 1028
		# PADDING = 0
		#
		# USE_FF = False
		# FF_LINEAR_BLOCK_SIZE = 256
		# FF_LINEAR_OUTPUT_SIZE = 256
		# FF_LINEAR_LAYERS = [256, 256]
		# FF_LINEAR_ACTIVATION = nn.ReLU()
		# FF_LINEAR_INIT = None
		# FF_LINEAR_NORM = [True] + [False for _ in FF_LINEAR_LAYERS]
		#
		# if USE_FF:
		# 	ff = LinearModel(
		# 		block_size=FF_LINEAR_BLOCK_SIZE,
		# 		vocab_size=FF_LINEAR_OUTPUT_SIZE,
		# 		dropout_rate=DROPOUT_RATE,
		# 		layer_sizes=FF_LINEAR_LAYERS,
		# 		hidden_activation=FF_LINEAR_ACTIVATION,
		# 		init_fn=FF_LINEAR_INIT,
		# 		norm=FF_LINEAR_NORM
		# 	)
		# else:
		# 	ff = None
		#
		# model = CNN(
		# 	extra_len=EXTRA_LEN,
		# 	num_classes=VOCAB_SIZE + 1,
		# 	conv_channels=CHANNELS,
		# 	kernel_sizes=KERNEL_SIZES,
		# 	hidden_activation=ACTIVATION,
		# 	pool_sizes=POOL_SIZES,
		# 	dropout_rate=DROPOUT_RATE,
		# 	padding=PADDING,
		# 	ff_linear=ff,
		# 	linear_collapse=True
		# )
		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/bemnetatlaw-drmca-cnn-111-experiment.zip")

		# DTYPE = torch.float32
		NP_DTYPE = np.float32
		X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/train/X/1712734175.835725.npy").astype(NP_DTYPE)
		y = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/train/y/1712734175.835725.npy").astype(NP_DTYPE)
		#
		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		self.assertEquals(y.shape, y_hat.shape)
