import unittest


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from core.utils.research.data.load.dataset import BaseDataset
from core.utils.research.model.model.decoder import Decoder
from core.utils.research.model.model.model import Transformer
from core.utils.research.training.trainer import Trainer


class TrainerTest(unittest.TestCase):

	def test_functionality(self):

		KERNEL_SIZE = 3
		BLOCK_SIZE = 1024
		EMB_SIZE = 64
		NUM_HEADS = 8
		FF_SIZE = 1024

		VOCAB_SIZE = 449
		BATCH_SIZE = 16

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

		dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/train"
			],
		)
		dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

		test_dataset = BaseDataset(
			[
				"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/test"
			],
		)
		dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

		loss_function = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=0.001)

		trainer = Trainer(model, loss_function=loss_function, optimizer=optimizer)
		trainer.train(dataloader, epochs=3, progress=True)

