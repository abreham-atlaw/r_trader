import unittest

import torch

from core.utils.research.model.model.decoder import Decoder
from core.utils.research.model.model.model import Transformer


class TransformerTest(unittest.TestCase):

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

		X = torch.rand((BATCH_SIZE, BLOCK_SIZE))
		y = model(X)

		self.assertEqual(y.shape, torch.Size((BATCH_SIZE, VOCAB_SIZE)))
