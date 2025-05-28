import hashlib
import os.path
import unittest

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.training.trainer import Trainer
from lib.utils.torch_utils.model_handler import ModelHandler
from test.core.utils.research.trainer.trainer_test import TrainerTest


class ModelHandlerTest(unittest.TestCase):

	@staticmethod
	def __create_model():

		CHANNELS = [128, 128] + [64 for _ in range(2)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 431
		POOL_SIZES = [3 for _ in CHANNELS]
		DROPOUT_RATE = 0.3
		ACTIVATION = nn.LeakyReLU()
		BLOCK_SIZE = 1024 + EXTRA_LEN
		PADDING = 0
		LINEAR_COLLAPSE = True
		FLATTEN_COLLAPSE = True
		AVG_POOL = True
		NORM = [True] + [False for _ in CHANNELS[1:]]
		LR = 1e-3

		INDICATORS_DELTA = True
		INDICATORS_SO = [14]
		INDICATORS_RSI = [14]

		USE_FF = True
		FF_LINEAR_BLOCK_SIZE = 64
		FF_LINEAR_OUTPUT_SIZE = 64
		FF_LINEAR_LAYERS = []
		FF_LINEAR_ACTIVATION = nn.ReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [True] + [False for _ in FF_LINEAR_LAYERS]
		FF_DROPOUT = 0.5

		BATCH_SIZE = 64

		if USE_FF:
			ff = LinearModel(
				block_size=FF_LINEAR_BLOCK_SIZE,
				vocab_size=FF_LINEAR_OUTPUT_SIZE,
				dropout_rate=FF_DROPOUT,
				layer_sizes=FF_LINEAR_LAYERS,
				hidden_activation=FF_LINEAR_ACTIVATION,
				init_fn=FF_LINEAR_INIT,
				norm=FF_LINEAR_NORM
			)
		else:
			ff = None

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI
		)

		model = CNN(
			extra_len=EXTRA_LEN,
			num_classes=VOCAB_SIZE + 1,
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
			input_size=BLOCK_SIZE
		)

		return model

	def test_save_and_load_integrity(self):

		SAVE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/model_handler_test/model.zip"

		model = self.__create_model()
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/rtrader-datapreparer-simsim-cum-0-it-2/trimmed"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)

		trainer = Trainer(model)
		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=1e-3)

		trainer.train(
			dataloader,
			epochs=5,
			progress=True,
			cls_loss_only=False
		)

		model.eval()
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/rtrader-datapreparer-simsim-cum-0-it-2/trimmed"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)
		with torch.no_grad():
			ins, outs = [], []
			for X, y in dataloader:
				X, y = X.to(trainer.device), y.to(trainer.device)
				ins.append(hashlib.md5(X.to('cpu').detach().numpy().tobytes()).hexdigest())
				outs.append(hashlib.md5(model(X).to('cpu').detach().numpy().tobytes()).hexdigest())
				if len(outs) >= 5:
					break
		model_hashes = hashlib.md5(''.join(ins).encode('utf-8')).hexdigest(), hashlib.md5(''.join(outs).encode('utf-8')).hexdigest()

		ModelHandler.save(model, SAVE_PATH)

		loaded_model = ModelHandler.load(SAVE_PATH)
		loaded_model.eval()

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_exports/rtrader-datapreparer-simsim-cum-0-it-2/trimmed"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)

		with torch.no_grad():
			ins, outs = [], []
			for X, y in dataloader:
				X, y = X.to('cpu'), y.to('cpu')
				ins.append(hashlib.md5(X.to('cpu').detach().numpy().tobytes()).hexdigest())
				outs.append(hashlib.md5(loaded_model(X).to('cpu').detach().numpy().tobytes()).hexdigest())
				if len(outs) >= 5:
					break
		loaded_hashes = hashlib.md5(''.join(ins).encode('utf-8')).hexdigest(), hashlib.md5(''.join(outs).encode('utf-8')).hexdigest()

		self.assertTrue(model_hashes, loaded_hashes)

		for X, y in dataloader:
			break

		ys = []
		for m in [model, loaded_model]:
			m.eval()
			with torch.no_grad():
				ys.append(m(X).detach().numpy())
		hashes = [hashlib.md5(y.tobytes()).hexdigest() for y in ys]

		self.assertTrue(np.all(ys[0] == ys[1]))
		self.assertTrue(hashes[0] == hashes[1])

	def test_load(self):
		LOCAL_PATH = os.path.abspath("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/model/test.zip")
		X = torch.from_numpy(np.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/6/train/X/1740913843.59131.npy").astype(np.float32))

		model = TrainerTest.create_cnn()
		model.eval()

		y = model(X)

		self.assertIsNotNone(model)

		ModelHandler.save(model, LOCAL_PATH)
		loaded = ModelHandler.load(LOCAL_PATH).eval()
		loaded.eval()

		y_hat = loaded(X)

		self.assertTrue(torch.allclose(y, y_hat))

