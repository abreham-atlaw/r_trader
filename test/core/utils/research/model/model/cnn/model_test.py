import unittest

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from core import Config
from core.agent.agents import TraderAgent
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.wrapped import WrappedModel
from lib.rl.agent.dta import TorchModel
from lib.utils.torch_utils.model_handler import ModelHandler
from temp import stats


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

		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-17-cum-0-it-2-tot_1.zip")

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
		# CHANNELS = [128, 128] + [64 for _ in range(2)]
		# EXTRA_LEN = 124
		# KERNEL_SIZES = [3 for _ in CHANNELS]
		# VOCAB_SIZE = 431
		# POOL_SIZES = [3 for _ in CHANNELS]
		# DROPOUT_RATE = 0.3
		# ACTIVATION = nn.LeakyReLU()
		# BLOCK_SIZE = 1024 + EXTRA_LEN
		# PADDING = 0
		# LINEAR_COLLAPSE = True
		# AVG_POOL = True
		# NORM = [True] + [False for _ in CHANNELS[1:]]
		# LR = 1e-3
		#
		# INDICATORS_DELTA = True
		# INDICATORS_SO = [14]
		# INDICATORS_RSI = [14]
		#
		# USE_FF = True
		# FF_LINEAR_LAYERS = [1024, 1024, VOCAB_SIZE + 1]
		# FF_LINEAR_ACTIVATION = nn.ReLU()
		# FF_LINEAR_INIT = None
		# FF_LINEAR_NORM = [True] + [False for _ in FF_LINEAR_LAYERS]
		# FF_DROPOUT = 0.5
		#
		# if USE_FF:
		# 	ff = LinearModel(
		# 		dropout_rate=FF_DROPOUT,
		# 		layer_sizes=FF_LINEAR_LAYERS,
		# 		hidden_activation=FF_LINEAR_ACTIVATION,
		# 		init_fn=FF_LINEAR_INIT,
		# 		norm=FF_LINEAR_NORM
		# 	)
		# else:
		# 	ff = None
		#
		# indicators = Indicators(
		# 	delta=INDICATORS_DELTA,
		# 	so=INDICATORS_SO,
		# 	rsi=INDICATORS_RSI
		# )
		#
		# model = CNN(
		# 	extra_len=EXTRA_LEN,
		# 	conv_channels=CHANNELS,
		# 	kernel_sizes=KERNEL_SIZES,
		# 	hidden_activation=ACTIVATION,
		# 	pool_sizes=POOL_SIZES,
		# 	dropout_rate=DROPOUT_RATE,
		# 	padding=PADDING,
		# 	avg_pool=AVG_POOL,
		# 	linear_collapse=LINEAR_COLLAPSE,
		# 	norm=NORM,
		# 	ff_block=ff,
		# 	indicators=indicators,
		# 	input_size=BLOCK_SIZE
		# )
		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-38-cum-0-it-2-tot.zip")

		# DTYPE = torch.float32
		NP_DTYPE = np.float32
		X = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/rtrader-datapreparer-simsim-cum-0-it-2/out/train/X/1725558208.724822.npy").astype(NP_DTYPE)
		y = np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/rtrader-datapreparer-simsim-cum-0-it-2/out/train/y/1725558208.724822.npy").astype(NP_DTYPE)
		#
		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		self.assertEquals(y.shape, y_hat.shape)

	def test_load_and_predict(self):

		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-7-cum-0-it-1-tot-2l.zip")
		model.eval()
		NP_DTYPE = np.float32

		X = np.load("/home/abrehamatlaw/Downloads/1722305017.596668.npy").astype(NP_DTYPE)

		with torch.no_grad():
			y = model(torch.from_numpy(X))

		self.assertIsNotNone(y)

	def test_plot_probability_distribution(self):

		agent = TraderAgent()

		node, repo = stats.load_node_repo("/home/abrehamatlaw/Downloads/Compressed/results_2/graph_dumps/1724040115.942755")
		state = repo.retrieve(node.id)
		inputs = agent._prepare_dra_input(state, node.children[0].action)

		for model_path in [
			"/home/abrehamatlaw/Downloads/Compressed/1723545682.854022.zip",
			"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-7-cum-0-it-1-tot.zip"
		]:
			model = TorchModel(
				WrappedModel(
					ModelHandler.load(model_path),
					seq_len=Config.MARKET_STATE_MEMORY,
					window_size=Config.AGENT_MA_WINDOW_SIZE
				)
			)
			out = model.predict(np.expand_dims(inputs, axis=0))

			prob_distribution = out[0, :-2]
			prob_distribution = (prob_distribution - np.min(prob_distribution)) / (
				np.sum(prob_distribution - np.min(prob_distribution)))

			plt.scatter(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND, prob_distribution)
		plt.show()
