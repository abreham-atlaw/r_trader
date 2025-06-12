import os
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.losses import CrossEntropyLoss, MeanSquaredErrorLoss
from core.utils.research.model.layers import Indicators, DynamicLayerNorm, DynamicBatchNorm, MinMaxNorm, Axis
from core.utils.research.model.model.cnn.bridge_block import BridgeBlock
from core.utils.research.model.model.cnn.cnn2 import CNN2
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.cnn.resnet.resnet_block import ResNetBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Transformer, DecoderBlock, TransformerEmbeddingBlock
from core.utils.research.training.trainer import Trainer
from core.utils.research.utils.model_migration.cnn_to_cnn2_migrator import CNNToCNN2Migrator
from lib.utils.torch_utils.model_handler import ModelHandler


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
		return self.create_cnn2()

	@staticmethod
	def create_cnn():
		CHANNELS = [128 for _ in range(4)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 431
		POOL_SIZES = [(0, 0.5, 3) for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = [nn.LeakyReLU(), nn.Identity(), nn.Identity(), nn.LeakyReLU()]
		BLOCK_SIZE = 1024 + EXTRA_LEN
		PADDING = 0
		LINEAR_COLLAPSE = True
		AVG_POOL = True
		NORM = [True] + [True for _ in CHANNELS[1:]]
		STRIDE = 2
		INPUT_DROPOUT = 0.2
		INPUT_NORM = False
		COLLAPSE_AVG_POOL = False
		LR = 1e-4

		POSITIONAL_ENCODING = True
		POSITIONAL_ENCODING_NORM = True
		INDICATORS_DELTA = True
		INDICATORS_SO = []
		INDICATORS_RSI = []
		INDICATORS_IDENTITIES = 4

		USE_CHANNEL_FFN = False
		CHANNEL_FFN_LAYERS = [CHANNELS[-1] for _ in range(4)]
		CHANNEL_FFN_DROPOUT = 0.1
		CHANNEL_FFN_ACTIVATION = nn.LeakyReLU()
		CHANNEL_FFN_NORM = [False] + [False for _ in CHANNEL_FFN_LAYERS[:-1]]
		CHANNEL_FFN_INIT = None

		USE_FF = True
		FF_LINEAR_LAYERS = [256 for _ in range(4)] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = nn.LeakyReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [DynamicBatchNorm()] + [MinMaxNorm() for _ in FF_LINEAR_LAYERS[:-1]]
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
			input_dropout=INPUT_DROPOUT,
			input_norm=INPUT_NORM,
			collapse_avg_pool=COLLAPSE_AVG_POOL,
		)
		return model

	@staticmethod
	def create_cnn2():
		CHANNELS = [32 for _ in range(4)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 431
		POOL_SIZES = [(0, 0.5, 3, 1) for _ in CHANNELS]
		DROPOUT_RATE = [0 for _ in CHANNELS]
		ACTIVATION = [nn.Identity(), nn.Identity(), nn.LeakyReLU(), nn.Identity()]
		BLOCK_SIZE = 1024 + EXTRA_LEN
		PADDING = 0
		NORM = [False] + [False for _ in CHANNELS[1:]]

		INDICATORS_DELTA = True
		INDICATORS_SO = []
		INDICATORS_RSI = []

		BRIDGE_FF_LINEAR_LAYERS = [512, 256]
		BRIDGE_FF_LINEAR_ACTIVATION = [nn.Identity() for _ in BRIDGE_FF_LINEAR_LAYERS]
		BRIDGE_FF_LINEAR_NORM = [DynamicLayerNorm() for _ in BRIDGE_FF_LINEAR_LAYERS]
		BRIDGE_FF_LINEAR_DROPOUT = 0

		COLLAPSE_INPUT_NORM = DynamicBatchNorm()
		DROPOUT_BRIDGE = 0.2

		FF_LINEAR_LAYERS = [64, 16] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = [nn.Identity(), nn.LeakyReLU()]
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [MinMaxNorm()] + [nn.Identity() for _ in FF_LINEAR_LAYERS[:-1]]
		FF_DROPOUT = 0

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI
		)

		return CNN2(
			extra_len=EXTRA_LEN,
			input_size=BLOCK_SIZE,

			embedding_block=EmbeddingBlock(
				indicators=indicators,
			),

			cnn_block=ResNetBlock(
				input_channels=indicators.indicators_len,
				conv_channels=CHANNELS,
				kernel_sizes=KERNEL_SIZES,
				pool_sizes=POOL_SIZES,
				hidden_activation=ACTIVATION,
				dropout_rate=DROPOUT_RATE,
				norm=NORM,
				padding=PADDING
			),

			bridge_block=BridgeBlock(
				ff_block=LinearModel(
					dropout_rate=BRIDGE_FF_LINEAR_DROPOUT,
					layer_sizes=BRIDGE_FF_LINEAR_LAYERS,
					hidden_activation=BRIDGE_FF_LINEAR_ACTIVATION,
					norm=BRIDGE_FF_LINEAR_NORM
				)
			),

			collapse_block=CollapseBlock(
				dropout=DROPOUT_BRIDGE,
				input_norm=COLLAPSE_INPUT_NORM,
				ff_block=LinearModel(
					dropout_rate=FF_DROPOUT,
					layer_sizes=FF_LINEAR_LAYERS,
					hidden_activation=FF_LINEAR_ACTIVATION,
					init_fn=FF_LINEAR_INIT,
					norm=FF_LINEAR_NORM
				)
			)

		)

	def __create_transformer(self):

		EXTRA_LEN = 124
		INPUT_SIZE = 1024 + EXTRA_LEN
		VOCAB_SIZE = 431

		# DECODER BLOCK
		EMBEDDING_CB_CHANNELS = [8]*2
		EMBEDDING_CB_KERNELS = [3]*len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_POOL_SIZES = [0] * len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_DROPOUTS = [0] * len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_NORM = [False] * len(EMBEDDING_CB_CHANNELS)
		EMBEDDING_CB_HIDDEN_ACTIVATION = nn.PReLU()
		INDICATORS_DELTA = True

		NUM_HEADS = 4

		# COLLAPSE BLOCK
		COLLAPSE_FF_LAYERS = [16] * 2 + [VOCAB_SIZE + 1]
		COLLAPSE_FF_DROPOUTS = [0] * (len(COLLAPSE_FF_LAYERS) - 1)
		COLLAPSE_FF_ACTIVATION = nn.PReLU()
		COLLAPSE_FF_NORM = [False] * len(COLLAPSE_FF_LAYERS)

		indicators = Indicators(
			delta=INDICATORS_DELTA,
		)

		return Transformer(
			input_size=INPUT_SIZE,
			extra_len=EXTRA_LEN,
			decoder_block=DecoderBlock(
				num_heads=NUM_HEADS,
				transformer_embedding_block=TransformerEmbeddingBlock(
					embedding_block=EmbeddingBlock(
						indicators=indicators
					),
					cnn_block=CNNBlock(
						input_channels=indicators.indicators_len,
						conv_channels=EMBEDDING_CB_CHANNELS,
						kernel_sizes=EMBEDDING_CB_KERNELS,
						pool_sizes=EMBEDDING_CB_POOL_SIZES,
						dropout_rate=EMBEDDING_CB_DROPOUTS,
						norm=EMBEDDING_CB_NORM,
						hidden_activation=EMBEDDING_CB_HIDDEN_ACTIVATION
					)
				),
			),

			collapse_block=CollapseBlock(
				ff_block=LinearModel(
					layer_sizes=COLLAPSE_FF_LAYERS,
					hidden_activation=COLLAPSE_FF_ACTIVATION,
					dropout_rate=COLLAPSE_FF_DROPOUTS,
					norm=COLLAPSE_FF_NORM
				),
			)
		)

	def __init_dataloader(self):
		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			check_file_sizes=True,
			load_weights=False,
		)
		dataloader = DataLoader(dataset, batch_size=64)

		test_dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test"
			],
			check_file_sizes=True,
			load_weights=False,
		)
		test_dataloader = DataLoader(test_dataset, batch_size=64)

		return dataloader, test_dataloader

	def __init_trainer(self, model):
		trainer = Trainer(model)
		trainer.cls_loss_function = CrossEntropyLoss(weighted_sample=False)
		trainer.reg_loss_function = MeanSquaredErrorLoss(weighted_sample=False)
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
			epochs=5,
			progress=True,
		)

		ModelHandler.save(self.trainer.model, SAVE_PATH)

		for X, y, w in self.test_dataloader:
			break

		loaded_model = ModelHandler.load(SAVE_PATH)
		loaded_model.eval()

		original_y = self.trainer.model(X)

		loaded_y = loaded_model(X)

		self.assertTrue(torch.allclose(original_y, loaded_y))

		self.trainer.validate(self.test_dataloader)

	def test_validate(self):
		self.trainer.validate(self.test_dataloader)
