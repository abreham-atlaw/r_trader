import unittest

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.training.trainer import Trainer
from core.utils.research.training.trainer.simple_trainer import SimpleTrainer
from lib.utils.torch_utils.model_handler import ModelHandler


class SimpleTrainerTest(unittest.TestCase):

	def setUp(self):
		CHANNELS = [128 for _ in range(4)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 431
		POOL_SIZES = [3 for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = nn.LeakyReLU()
		BLOCK_SIZE = 1024 + EXTRA_LEN
		PADDING = 0
		LINEAR_COLLAPSE = True
		AVG_POOL = True
		NORM = [False] + [False for _ in CHANNELS[1:]]
		STRIDE = 2
		LR = 1e-4

		POSITIONAL_ENCODING = True
		POSITIONAL_ENCODING_NORM = True
		INDICATORS_DELTA = True
		INDICATORS_SO = []
		INDICATORS_RSI = []
		INDICATORS_IDENTITIES = 4

		USE_FF = True
		FF_LINEAR_LAYERS = [256 for _ in range(4)] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = nn.LeakyReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [False] + [False for _ in FF_LINEAR_LAYERS[:-1]]
		FF_NORM_LEARNABLE = False
		FF_DROPOUT = 0.12

		if USE_FF:
			ff = LinearModel(
				dropout_rate=FF_DROPOUT,
				layer_sizes=FF_LINEAR_LAYERS,
				hidden_activation=FF_LINEAR_ACTIVATION,
				init_fn=FF_LINEAR_INIT,
				norm=FF_LINEAR_NORM,
				norm_learnable=FF_NORM_LEARNABLE
			)
		else:
			ff = None

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI,
			identities=INDICATORS_IDENTITIES,
		)

		model = CNN(
			extra_len=EXTRA_LEN,
			conv_channels=CHANNELS,
			kernel_sizes=KERNEL_SIZES,
			hidden_activation=ACTIVATION,
			pool_sizes=POOL_SIZES,
			dropout_rate=DROPOUT_RATE,
			padding=PADDING,
			avg_pool=AVG_POOL,
			linear_collapse=LINEAR_COLLAPSE,
			norm=NORM,
			ff_block=ff,
			indicators=indicators,
			input_size=BLOCK_SIZE,
			positional_encoding=POSITIONAL_ENCODING,
			norm_positional_encoding=POSITIONAL_ENCODING_NORM,
			stride=STRIDE
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
			check_file_sizes=True
		)
		self.dataloader = DataLoader(dataset, batch_size=8)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback(
			# 	"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new",
			# 	save_state=True
			# ),
			# WeightStatsCallback()
		]

		self.trainer = SimpleTrainer(model)
		self.trainer.cls_loss_function = nn.CrossEntropyLoss()
		self.trainer.reg_loss_function = nn.MSELoss()
		self.trainer.optimizer = Adam(self.trainer.model.parameters(), lr=LR)

	def test_train(self):
		self.trainer.train(
			self.dataloader,
			epochs=10,
		)
