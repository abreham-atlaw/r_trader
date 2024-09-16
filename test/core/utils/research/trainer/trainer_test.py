import unittest


import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from core import Config
from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.losses import WeightedCrossEntropyLoss, WeightedMSELoss
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer
from core.utils.research.training.callbacks import WeightStatsCallback
from core.utils.research.training.callbacks.checkpoint_callback import CheckpointCallback
from core.utils.research.training.callbacks.metric_callback import MetricCallback
from core.utils.research.training.data.repositories.metric_repository import MetricRepository, MongoDBMetricRepository
from core.utils.research.training.data.state import TrainingState
from core.utils.research.training.trainer import Trainer
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

	def test_cnn_model(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.zip"

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
		AVG_POOL = True
		NORM = [True] + [True for _ in CHANNELS[1:]]
		LR = 1e-3

		INDICATORS_DELTA = True
		INDICATORS_SO = [14]
		INDICATORS_RSI = [14]

		USE_FF = True
		FF_LINEAR_LAYERS = [1024, 1024, VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = nn.ReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [True] + [False for _ in FF_LINEAR_LAYERS[:-1]]
		FF_DROPOUT = 0.5

		if USE_FF:
			ff = LinearModel(
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
		# model = LinearModel(
		# 	block_size=1028,
		# 	vocab_size=432,
		# 	dropout_rate=0.0,
		# 	layer_sizes=[
		# 		64,
		# 		64,
		# 	]
		# )

		# ModelHandler.save(model, SAVE_PATH)

		dataset = BaseDataset(
			[
				"/home/abrehamatlaw/Downloads/Compressed/out_1/kaggle/input/rtrader-datapreparer-simsim-cum-0-it-2/out/train"
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

		trainer = Trainer(model)
		# trainer.cls_loss_function = WeightedMSELoss(VOCAB_SIZE-1, softmax=True)
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
