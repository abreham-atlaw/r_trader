import unittest


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer
from core.utils.research.training.callbacks.checkpoint_callback import CheckpointCallback
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

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/actual_cnn_model.pth"

		VOCAB_SIZE = 449
		BATCH_SIZE = 4
		CHANNELS = [1, 128, 256]
		KERNEL_SIZES = [3, 3]
		POOL_SIZES = [1, 1]
		DROPOOUTS = 0.1

		# model = CNN(
		# 	VOCAB_SIZE,
		# 	conv_channels=CHANNELS,
		# 	kernel_sizes=KERNEL_SIZES,
		# 	pool_sizes=POOL_SIZES,
		# 	dropout_rate=DROPOOUTS
		# )
		model = LinearModel(
			block_size=1024,
			vocab_size=449,
			dropout_rate=0.1,
			layer_sizes=[
				128,
				256,
			]
		)

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

		callbacks = [
			CheckpointCallback("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/raw/new", save_state=True)
		]

		trainer = Trainer(model, callbacks=callbacks)
		trainer.loss_function = nn.CrossEntropyLoss()
		trainer.optimizer = Adam(trainer.model.parameters(), lr=1e-3)

		# trainer.train(dataloader, epochs=5, progress=True, val_dataloader=test_dataloader)
		ModelHandler.save(trainer.model, SAVE_PATH)
		torch.save(model.state_dict(), SAVE_PATH)

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

		trainer = Trainer(model, loss_function=loss_function, optimizer=optimizer)
		trainer.train(dataloader, epochs=2, progress=True, val_dataloader=test_dataloader)
		torch.save(model.state_dict(), SAVE_PATH)

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
		trainer.loss_function = nn.MSELoss()
		trainer.optimizer = Adam(trainer.model.parameters())

		trainer.train(
			dataloader,
			epochs=15
		)

