import os
import unittest
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.data.load.ensemble import EnsembleStackedDataset
from core.utils.research.losses import CrossEntropyLoss, MeanSquaredErrorLoss
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.cnn.resnet import ResNet
from core.utils.research.model.model.ensemble.stacked import LinearMSM, SimplifiedMSM, PlainMSM
from core.utils.research.model.model.ensemble.stacked.linear_3dmsm import Linear3dMSM
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer
from core.utils.research.training.callbacks import WeightStatsCallback
from core.utils.research.training.callbacks.checkpoint_callback import CheckpointCallback
from core.utils.research.training.callbacks.metric_callback import MetricCallback
from core.utils.research.training.data.repositories.metric_repository import MetricRepository, MongoDBMetricRepository
from core.utils.research.training.data.state import TrainingState
from core.utils.research.training.trackers.stats_tracker import DynamicStatsTracker, Keys, WeightsStatsTracker, \
	GradientsStatsTracker
from core.utils.research.training.trainer import Trainer
from lib.utils.torch_utils.model_handler import ModelHandler
from lib.utils.torch_utils.tensor_merger import TensorMerger


class SineWaveDataset(Dataset):
	def __init__(self, total_samples, sequence_length, resolution=500):
		self.total_samples = total_samples
		self.sequence_length = sequence_length
		self.resolution = resolution
		self.data = np.sin(np.linspace(0, 2 * np.pi * resolution, total_samples + sequence_length))

	def __len__(self):
		return self.total_samples

	def shuffle(self):
		np.random.shuffle(self.data)

	def __getitem__(self, idx):
		start = idx
		end = start + self.sequence_length
		sequence = self.data[start:end]
		next_value = np.expand_dims(self.data[end], axis=0)
		return torch.tensor(sequence, dtype=torch.float32), torch.tensor(next_value, dtype=torch.float32)


class TrainerTest(unittest.TestCase):

	def __generate_dataset(self, sample_path: str, target_path: str, size: int):
		print(f"Generating {target_path}...")

		files = [os.path.join(sample_path, filename) for filename in os.listdir(sample_path)]

		if os.path.isdir(files[0]):
			for directory in files:
				self.__generate_dataset(directory, os.path.join(target_path, os.path.basename(directory)), size)
			return

		os.makedirs(target_path)
		target_shape = np.load(files[0]).shape

		for i in range(size):
			array = np.random.random(target_shape)
			np.save(os.path.join(target_path, f"{i}.npy"), array)
			print(f"Generated: {(i+1)*100/size:.2f}%")

		print(f"Generated: {target_path}")

	def __create_model(self):
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
		INPUT_DROPOUT = 0.2
		LR = 1e-4

		POSITIONAL_ENCODING = True
		POSITIONAL_ENCODING_NORM = True
		INDICATORS_DELTA = True
		INDICATORS_SO = []
		INDICATORS_RSI = []
		INDICATORS_IDENTITIES = 4

		USE_CHANNEL_FFN = True
		CHANNEL_FFN_LAYERS = [CHANNELS[-1] for _ in range(4)]
		CHANNEL_FFN_DROPOUT = 0.1
		CHANNEL_FFN_ACTIVATION = nn.LeakyReLU()
		CHANNEL_FFN_NORM = [False] + [False for _ in CHANNEL_FFN_LAYERS[:-1]]
		CHANNEL_FFN_INIT = None

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

		if USE_CHANNEL_FFN:
			channel_ffn = LinearModel(
				dropout_rate=CHANNEL_FFN_DROPOUT,
				layer_sizes=CHANNEL_FFN_LAYERS,
				hidden_activation=CHANNEL_FFN_ACTIVATION,
				init_fn=CHANNEL_FFN_INIT,
				norm=CHANNEL_FFN_NORM
			)
		else:
			channel_ffn = None

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
			stride=STRIDE,
			channel_ffn=channel_ffn,
			input_dropout=INPUT_DROPOUT
		)
		return model

	def __init_dataloader(self):
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			check_file_sizes=True,
			load_weights=True,
		)
		dataloader = DataLoader(dataset, batch_size=64)

		test_dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			check_file_sizes=True,
			load_weights=True,
		)
		test_dataloader = DataLoader(test_dataset, batch_size=64)

		return dataloader, test_dataloader

	def __init_trainer(self, model):
		trainer = Trainer(model)
		trainer.cls_loss_function = CrossEntropyLoss(weighted_sample=False)
		trainer.reg_loss_function = MeanSquaredErrorLoss(weighted_sample=True)
		trainer.optimizer = Adam(trainer.model.parameters())
		return trainer

	def setUp(self):
		self.model = self.__create_model()
		self.dataloader, self.test_dataloader = self.__init_dataloader()
		self.trainer = self.__init_trainer(self.model)

	def test_train(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip"

		self.trainer.train(
			self.dataloader,
			val_dataloader=self.test_dataloader,
			epochs=10,
			progress=True,
		)

		ModelHandler.save(self.trainer.model, SAVE_PATH)

	def test_validate(self):
		self.trainer.validate(self.test_dataloader)
