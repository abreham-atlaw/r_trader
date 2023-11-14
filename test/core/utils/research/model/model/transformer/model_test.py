import unittest

import numpy as np
import torch

from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer


class TransformerTest(unittest.TestCase):

	def test_functionality(self):
		KERNEL_SIZE = 3
		BLOCK_SIZE = 1024
		EMB_SIZE = 8
		NUM_HEADS = 2
		FF_SIZE = 128

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

		X = torch.rand((BATCH_SIZE, BLOCK_SIZE))
		with torch.no_grad():
			y = model(X)

		y_classes = np.argmax(y.detach().numpy(), axis=1)

		self.assertEqual(y.shape, torch.Size((BATCH_SIZE, VOCAB_SIZE)))

	def test_load_and_run(self):
		KERNEL_SIZE = 3
		BLOCK_SIZE = 1024
		EMB_SIZE = 128
		NUM_HEADS = 64
		FF_SIZE = 254
		VOCAB_SIZE = 449

		DTYPE = torch.float32
		NP_DTYPE = np.float32

		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train/X/1699605237.250606.npy").astype(NP_DTYPE)[:20]
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual/train/y/1699605237.250606.npy").astype(NP_DTYPE)[:20]

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
		model.load_state_dict(torch.load('/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/actual_model.pth', map_location=torch.device('cpu')))
		model.eval()

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		y_hat_classes = np.argmax(y_hat.detach().numpy(), axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertEqual(y_hat_classes.shape, (X.shape[0],))
