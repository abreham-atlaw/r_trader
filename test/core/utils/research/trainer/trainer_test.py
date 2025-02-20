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
from core.utils.research.losses import WeightedCrossEntropyLoss, WeightedMSELoss, MeanSquaredClassError, \
	MSCECrossEntropyLoss, LogLoss
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.ensemble.stacked.msm import LinearMSM, SimplifiedMSM, PlainMSM
from core.utils.research.model.model.ensemble.stacked.msm.linear_3dmsm import Linear3dMSM

from core.utils.research.model.model.ensemble.stacked.encdec_fusion import EncDecFusionModel
from core.utils.research.model.model.ensemble.stacked.simplified_sem import SimplifiedSEM
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Transformer, DecoderBlock, TransformerEmbeddingBlock
from core.utils.research.training.callbacks import WeightStatsCallback
from core.utils.research.training.callbacks.checkpoint_callback import CheckpointCallback, StoreCheckpointCallback
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

	def setUp(self):

		SAMPLE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
		ENSEMBLE_SAMPLE_PATHES = [
				# "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/model_output/cnn-192",
				# "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/model_output/cnn-240"
		]

		SAMPLE_PATHS = [SAMPLE_PATH] + ENSEMBLE_SAMPLE_PATHES

		GENERATION_SIZE = 100

		self.GENERATED_PATHS_MAP = {
			path: f"{path}_{GENERATION_SIZE}" for path in SAMPLE_PATHS
		}

		self.GENERATED_PATH = self.GENERATED_PATHS_MAP[SAMPLE_PATH]
		self.ENSEMBLE_GENERATED_PATH = [self.GENERATED_PATHS_MAP[path] for path in ENSEMBLE_SAMPLE_PATHES]

		for sample_path, target_path in self.GENERATED_PATHS_MAP.items():
			target_path = f"{sample_path}_{GENERATION_SIZE}"
			if os.path.exists(target_path):
				continue
			self.__generate_dataset(sample_path, target_path, GENERATION_SIZE)

	def test_linear(self):
		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/linear_test.zip"

		VOCAB_SIZE = 431
		DROPOUT = 0.5
		LAYER_SIZES = [128, 128, VOCAB_SIZE + 1]
		HIDDEN_ACTIVATION = nn.LeakyReLU()
		INIT_FUNCTION = None
		NORM = [True] + [False for _ in LAYER_SIZES[1:]]
		BLOCK_SIZE = 1148
		LR = 1e-3

		model = LinearModel(
			dropout_rate=DROPOUT,
			layer_sizes=LAYER_SIZES,
			hidden_activation=HIDDEN_ACTIVATION,
			init_fn=INIT_FUNCTION,
			norm=NORM,
			input_size=BLOCK_SIZE
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)

		dataloader = DataLoader(dataset, batch_size=8)

		trainer = Trainer(
			model,
		)
		trainer.cls_loss_function = MSCECrossEntropyLoss(
			classes=np.array(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND),
			epsilon=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON,
			device=trainer.device,
			weights=[0.1, 0.9]
		)
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=LR)

		trainer.train(
			dataloader,
			epochs=10,
			progress=True,
			cls_loss_only=False
		)

		ModelHandler.save(trainer.model, SAVE_PATH)

	def test_cnn_model(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip"

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

		COLLAPSE_DROPOUT = 0.3

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
			embedding_block=EmbeddingBlock(
				positional_encoding=POSITIONAL_ENCODING,
				norm_positional_encoding=POSITIONAL_ENCODING_NORM,
				indicators=indicators,
			),
			cnn_block=CNNBlock(
				input_channels=indicators.indicators_len,
				conv_channels=CHANNELS,
				kernel_sizes=KERNEL_SIZES,
				hidden_activation=ACTIVATION,
				pool_sizes=POOL_SIZES,
				dropout_rate=DROPOUT_RATE,
				padding=PADDING,
				avg_pool=AVG_POOL,
				norm=NORM,
				stride=STRIDE
			),
			collapse_block=CollapseBlock(
				ff_block=ff,
				dropout=COLLAPSE_DROPOUT
			),
			input_size=BLOCK_SIZE,
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
			check_file_sizes=True
		)
		dataloader = DataLoader(dataset, batch_size=64)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
			# WeightStatsCallback()
		]

		trainer = Trainer(model)
		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=LR)

		trainer.train(
			dataloader,
			epochs=10,
			progress=True,
			cls_loss_only=False
		)

		ModelHandler.save(trainer.model, SAVE_PATH)

	def test_resnet_model(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip"

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

		model = ResNet(
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
		dataloader = DataLoader(dataset, batch_size=8)

		trainer = Trainer(model)
		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=LR)

		trainer.train(
			dataloader,
			epochs=10,
			progress=True,
			cls_loss_only=False
		)

		ModelHandler.save(trainer.model, SAVE_PATH)

	def test_transformer(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip"

		CHANNELS = [128 for _ in range(4)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 431
		POOL_SIZES = [3 for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = nn.LeakyReLU()
		BLOCK_SIZE = 1024 + EXTRA_LEN
		PADDING = 0
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

		COLLAPSE_DROPOUT = 0.3

		USE_FF = True
		FF_LINEAR_LAYERS = [256 for _ in range(4)] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = nn.LeakyReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [False] + [False for _ in FF_LINEAR_LAYERS[:-1]]
		FF_NORM_LEARNABLE = False
		FF_DROPOUT = 0.12

		ATTN_NUM_HEADS = 4

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

		model = Transformer(
			extra_len=EXTRA_LEN,
			decoder_block=DecoderBlock(
				transformer_embedding_block=TransformerEmbeddingBlock(
					embedding_block=EmbeddingBlock(
						positional_encoding=POSITIONAL_ENCODING,
						norm_positional_encoding=POSITIONAL_ENCODING_NORM,
						indicators=indicators,
					),
					cnn_block=CNNBlock(
						input_channels=indicators.indicators_len,
						conv_channels=CHANNELS,
						kernel_sizes=KERNEL_SIZES,
						hidden_activation=ACTIVATION,
						pool_sizes=POOL_SIZES,
						dropout_rate=DROPOUT_RATE,
						padding=PADDING,
						avg_pool=AVG_POOL,
						norm=NORM,
						stride=STRIDE
					),
				),
				num_heads=ATTN_NUM_HEADS
			),
			collapse_block=CollapseBlock(
				ff_block=ff,
				dropout=COLLAPSE_DROPOUT
			),
			input_size=BLOCK_SIZE,
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
			check_file_sizes=True
		)
		dataloader = DataLoader(dataset, batch_size=8)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
			# WeightStatsCallback()
		]

		trainer = Trainer(model)
		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=LR)

		trainer.train(
			dataloader,
			epochs=10,
			progress=True,
			cls_loss_only=False
		)

		ModelHandler.save(trainer.model, SAVE_PATH)

	def __train_model(self, model, dataloader, callbacks=None):

		if callbacks is None:
			callbacks = []

		SAVE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/ensemble_stacked.zip"
		LR = 1e-3

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		# callbacks = [
		# 	# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
		# 	# WeightStatsCallback()
		# ]

		trainer = Trainer(model, callbacks=callbacks)
		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=LR)

		trainer.train(
			dataloader,
			epochs=10,
			progress=True,
			cls_loss_only=False,
		)

		ModelHandler.save(trainer.model, SAVE_PATH)

	def test_linear_msm(self):

		MODELS = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-240-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-192-cum-0-it-4-tot.zip",
			]
		]

		X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1724671615.45445.npy").astype(
			np.float32)

		model = LinearMSM(
			models=MODELS,
			ff=LinearModel(
				layer_sizes=[512, 256]
			)
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
			# WeightStatsCallback()
		]

		self.__train_model(model, dataloader)

	def test_linear_3d_msm(self):

		MODELS = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-240-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-192-cum-0-it-4-tot.zip",
			]
		]

		X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1724671615.45445.npy").astype(
			np.float32)

		model = Linear3dMSM(
			models=MODELS,
			ff=LinearModel(
				layer_sizes=[512, 256]
			)
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
			# WeightStatsCallback()
		]

		self.__train_model(model, dataloader)

	def test_plain_msm(self):

		MODELS = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-240-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-192-cum-0-it-4-tot.zip",
			]
		]

		X = np.load(
			"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1724671615.45445.npy").astype(
			np.float32)

		model = PlainMSM(
			models=MODELS,
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
			# WeightStatsCallback()
		]

		self.__train_model(model, dataloader)

	def test_simplified_msm(self):

		MODELS = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-6-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip"
			]
		]

		# X = np.load(
		# 	"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train/X/1724671615.45445.npy").astype(
		# 	np.float32)

		# model = LinearMSM(
		# 	models=MODELS,
		# 	ff=LinearModel(
		# 		layer_sizes=[512, 256]
		# 	)
		# )

		model = PlainMSM(
			models=MODELS
		)

		dataset = BaseDataset(
			root_dirs=[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/model_output/cnn-148.cnn-168"
			],
			check_last_file=True,
		)
		dataloader = DataLoader(dataset, batch_size=8)

		merger = TensorMerger()
		merger.load_config(os.path.join(dataset.root_dirs[0], "merger.pkl"))

		model = SimplifiedMSM(model=model, merger=merger)

		# test_dataset = BaseDataset(
		# 	[
		# 		"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/test"
		# 	],
		# )
		# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		callbacks = [
			# CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True),
			# WeightStatsCallback()
		]

		self.__train_model(model, dataloader)

	def test_encdec_fusion(self):
		models = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-240-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-192-cum-0-it-4-tot.zip",
			]
		]

		# ENCODER CONFIGS
		ENC_CHANNELS = [64 for _ in range(2)]
		ENC_KERNEL_SIZES = [3 for _ in ENC_CHANNELS]
		ENC_POOL_SIZES = [0 for _ in ENC_CHANNELS]
		ENC_DROPOUT_RATE = 0
		ENC_ACTIVATION = nn.LeakyReLU()
		ENC_PADDING = 0
		ENC_NORM = [False] + [False for _ in ENC_CHANNELS[1:]]

		ENC_INDICATORS_DELTA = True
		ENC_INDICATORS_SO = []
		ENC_INDICATORS_RSI = []

		ENC_NUM_HEADS = 4

		ENC_SEQUENCE_ONLY_ENCODER = True
		ENC_EXTRA_LEN = 124

		# DECODER CONFIGS
		DEC_CHANNELS = [64 for _ in range(2)]
		DEC_KERNEL_SIZES = [3 for _ in DEC_CHANNELS]
		DEC_POOL_SIZES = [0 for _ in DEC_CHANNELS]
		DEC_DROPOUT_RATE = 0
		DEC_ACTIVATION = nn.LeakyReLU()
		DEC_PADDING = 0
		DEC_NORM = [False] + [False for _ in DEC_CHANNELS[1:]]

		DEC_NUM_HEADS = 4

		# CHANNEL CONFIGS
		CHANNEL_INPUT_CHANNELS = ENC_CHANNELS[-1] + DEC_CHANNELS[-1]
		CHANNEL_CHANNELS = [64 for _ in range(2)]
		CHANNEL_KERNEL_SIZES = [3 for _ in CHANNEL_CHANNELS]
		CHANNEL_POOL_SIZES = [0 for _ in CHANNEL_CHANNELS]
		CHANNEL_DROPOUT_RATE = 0
		CHANNEL_ACTIVATION = nn.LeakyReLU()
		CHANNEL_PADDING = 0
		CHANNEL_NORM = [False] + [False for _ in DEC_CHANNELS[1:]]

		# FF Configs
		FF_LINEAR_LAYERS = [1024 for _ in range(2)]
		FF_LINEAR_ACTIVATION = nn.LeakyReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [False] + [False for _ in FF_LINEAR_LAYERS[:-1]]
		FF_DROPOUT = 0

		# ENCDECFUSION Configs
		PRECONCAT_NORM = True
		ENCODER_DROPOUT = 0.3
		DECODER_DROPOUT = 0.3

		enc_indicators = Indicators(
			delta=ENC_INDICATORS_DELTA,
			so=ENC_INDICATORS_SO,
			rsi=ENC_INDICATORS_RSI
		)

		model = EncDecFusionModel(
			encoder=DecoderBlock(
				transformer_embedding_block=TransformerEmbeddingBlock(
					embedding_block=EmbeddingBlock(
						indicators=enc_indicators,
					),
					cnn_block=CNNBlock(
						input_channels=enc_indicators.indicators_len,
						conv_channels=ENC_CHANNELS,
						kernel_sizes=ENC_KERNEL_SIZES,
						pool_sizes=ENC_POOL_SIZES,
						hidden_activation=ENC_ACTIVATION,
						dropout_rate=ENC_DROPOUT_RATE,
						norm=ENC_NORM,
						padding=ENC_PADDING,
					)
				),
				num_heads=ENC_NUM_HEADS,
				embedding_last=True
			),
			decoder=DecoderBlock(
				transformer_embedding_block=TransformerEmbeddingBlock(
					cnn_block=CNNBlock(
						input_channels=len(models),
						conv_channels=DEC_CHANNELS,
						kernel_sizes=DEC_KERNEL_SIZES,
						pool_sizes=DEC_POOL_SIZES,
						hidden_activation=DEC_ACTIVATION,
						dropout_rate=DEC_DROPOUT_RATE,
						norm=DEC_NORM,
						padding=DEC_PADDING,
					)
				),
				num_heads=DEC_NUM_HEADS,
				embedding_last=True
			),
			channel_block=CNNBlock(
				input_channels=CHANNEL_INPUT_CHANNELS,
				conv_channels=CHANNEL_CHANNELS,
				kernel_sizes=CHANNEL_KERNEL_SIZES,
				pool_sizes=CHANNEL_POOL_SIZES,
				hidden_activation=CHANNEL_ACTIVATION,
				dropout_rate=CHANNEL_DROPOUT_RATE,
				norm=CHANNEL_NORM,
				padding=CHANNEL_PADDING
			),
			collapse_block=CollapseBlock(
				ff_block=LinearModel(
					dropout_rate=FF_DROPOUT,
					layer_sizes=FF_LINEAR_LAYERS,
					hidden_activation=FF_LINEAR_ACTIVATION,
					init_fn=FF_LINEAR_INIT,
					norm=FF_LINEAR_NORM
				),
				extra_mode=False
			),
			models=models,
			preconcat_norm=PRECONCAT_NORM,
			sequence_only_encoding=ENC_SEQUENCE_ONLY_ENCODER,
			encoder_extra_len=ENC_EXTRA_LEN,
			encoder_dropout=ENCODER_DROPOUT,
			decoder_dropout=DECODER_DROPOUT
		)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=8)

		self.__train_model(model, dataloader)

	def test_simplified_sem(self):
		MODELS = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-6-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-168-cum-0-it-4-tot.zip"
			]
		]

		models = [
			ModelHandler.load(path)
			for path in [
				"/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-240-cum-0-it-4-tot.zip",
				"/home/abrehamatlaw/Downloads/Compressed/results_1/abrehamalemu-rtrader-training-exp-0-cnn-192-cum-0-it-4-tot.zip",
			]
		]

		CHANNELS = [128 for _ in range(4)]
		EXTRA_LEN = 124
		KERNEL_SIZES = [3 for _ in CHANNELS]
		POOL_SIZES = [3 for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = nn.LeakyReLU()
		PADDING = 0
		NORM = [False] + [False for _ in CHANNELS[1:]]

		model = EncDecFusionModel(
			encoder=LinearModel(
				layer_sizes=[512, 256]
			),
			decoder=EDFDecoder(
				channel_block=CNNBlock(
					input_channels=len(models),
					conv_channels=CHANNELS,
					kernel_sizes=KERNEL_SIZES,
					pool_sizes=POOL_SIZES,
					dropout_rate=DROPOUT_RATE,
					hidden_activation=ACTIVATION,
					norm=NORM,
					padding=PADDING,
				),
				ff_block=LinearModel(
					layer_sizes=[512, 256]
				)
			),
			collapse_block=LinearModel(
				layer_sizes=[512, 256]
			),
			models=models
		)

		dataset = BaseDataset(
			root_dirs=[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/model_output/cnn-148.cnn-168"
			],
			check_last_file=True,
		)
		dataloader = DataLoader(dataset, batch_size=8)

		SAVE_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/sem_model.zip"
		callbacks = [
			StoreCheckpointCallback(
				path=SAVE_PATH,
				active=True,
				simplified_mode=True
			)
		]

		merger = TensorMerger()
		merger.load_config(os.path.join(dataset.root_dirs[0], "merger.pkl"))

		model = SimplifiedSEM(model=model, merger=merger)

		self.__train_model(model, dataloader, callbacks=callbacks)

	def test_functionality(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/actual_model.pth"

		KERNEL_SIZE = 3
		BLOCK_SIZE = 1024
		EMB_SIZE = 8
		NUM_HEADS = 2
		FF_SIZE = 128

		VOCAB_SIZE = 449
		BATCH_SIZE = 4

		model = Transformer(
			Decoder(
				kernel_size=KERNEL_SIZE,
				emb_size=EMB_SIZE,
				block_size=BLOCK_SIZE,
				num_heads=NUM_HEADS,
				ff_size=FF_SIZE
			),
			vocab_size=VOCAB_SIZE
		)
		# model.load_state_dict(
		# 	torch.load('/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/sin_model.pth',
		# 	map_location=torch.device('cpu'))
		# )

		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

		test_dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/test"
			],
		)
		test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		loss_function = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=1e-3)

		trainer = Trainer(
			model,
			cls_loss_function=loss_function,
			optimizer=optimizer)
		trainer.train(dataloader, epochs=2, progress=True)
		torch.save(model.state_dict(), SAVE_PATH)

	def test_resume(self):

		CHANNELS = [128, 128]
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = 51
		POOL_SIZES = [0 for _ in CHANNELS]
		DROPOUT_RATE = 0
		ACTIVATION = nn.ReLU()
		BATCH_SIZE = 64

		state = TrainingState(
			id="test_id",
			epoch=5,
			batch=0
		)

		model = CNN(
			num_classes=VOCAB_SIZE,
			conv_channels=CHANNELS,
			kernel_sizes=KERNEL_SIZES,
			hidden_activation=ACTIVATION,
			pool_sizes=POOL_SIZES,
			dropout_rate=DROPOUT_RATE
		)

		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared(64)/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

		trainer = Trainer(
			model,
			callbacks=[
				MetricCallback(
					MongoDBMetricRepository(
						Config.MONGODB_URL,
						state.id
					)
				)
			]
		)

		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.optimizer = Adam(model.parameters(), lr=1e-3)

		trainer.train(dataloader, epochs=10, progress=True, state=state, val_dataloader=dataloader)

		new_state = trainer.state
		self.assertIsNotNone(new_state)

	def test_validate(self):

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
			check_file_sizes=True
		)
		dataloader = DataLoader(dataset, batch_size=8)

		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-148-cum-0-it-6-tot.zip")

		trainer = Trainer(model)
		trainer.cls_loss_function = nn.CrossEntropyLoss()
		trainer.reg_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(model.parameters(), lr=1e-3)

		print("Loss", trainer.validate(dataloader))
	# 	[10.343830108642578, 0.0007124464027583599, 10.34454345703125]

	def test_sinwave_prediction(self):

		dataset = SineWaveDataset(10000, 64)
		dataloader = DataLoader(dataset, batch_size=8)

		model = LinearModel(
			block_size=1024,
			vocab_size=449,
			dropout_rate=0.1,
			layer_sizes=[
				128,
				256,
			]
		)

		trainer = Trainer(model)
		trainer.cls_loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters())

		trainer.train(
			dataloader,
			epochs=15
		)

	def test_summary(self):
		model = ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/1723549677.105747.zip")
		trainer = Trainer(model)
		trainer.summary()

		self.assertIsNotNone(model)

