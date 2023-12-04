import unittest


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer
from core.utils.research.training.trainer import Trainer


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

		loss_function = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=1e-3)

		trainer = Trainer(model, loss_function=loss_function, optimizer=optimizer)
		trainer.train(dataloader, epochs=2, progress=True, val_dataloader=test_dataloader)
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

