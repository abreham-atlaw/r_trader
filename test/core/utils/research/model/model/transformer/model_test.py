import unittest

import numpy as np
import torch
from torch import nn

from core.utils.research.model.layers import Indicators
from core.utils.research.model.layers.cnn_block import CNNBlock
from core.utils.research.model.layers.collapse_ff_block import CollapseFFBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Decoder
from core.utils.research.model.model.transformer import Transformer
from lib.utils.torch_utils.model_handler import ModelHandler


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
				input_size=BLOCK_SIZE,
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
		NP_DTYPE = np.float32
		X = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/train/X/1712734175.835725.npy").astype(NP_DTYPE)[:20]
		y = np.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/train/y/1712734175.835725.npy").astype(NP_DTYPE)[:20]

		model = ModelHandler.load("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra-transformer.zip")

		with torch.no_grad():
			y_hat: torch.Tensor = model(torch.from_numpy(X))

		y_hat_classes = np.argmax(y_hat.detach().numpy(), axis=1)
		y_classes = np.argmax(y, axis=1)

		self.assertEqual(y_hat_classes.shape, (X.shape[0],))

	def test_save_load_integrity(self):

		SAVE_PATH = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/models/dra.0.zip"

		CHANNELS = [432]
		EXTRA_LEN = 4
		KERNEL_SIZES = [3]
		VOCAB_SIZE = 431
		POOL_SIZES = [0]
		DROPOUT_RATE = 0
		ACTIVATION = nn.Identity()
		BLOCK_SIZE = 1028
		PADDING = 0
		LINEAR_COLLAPSE = True
		AVG_POOL = True
		NORM = [True] + [False for _ in CHANNELS[1:]]

		USE_FF = False
		FF_LINEAR_BLOCK_SIZE = 256
		FF_LINEAR_OUTPUT_SIZE = 256
		FF_LINEAR_LAYERS = []
		FF_LINEAR_ACTIVATION = nn.ReLU()
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [True] + [False for _ in FF_LINEAR_LAYERS]

		INDICATOR = Indicators(
			delta=True,
			so=[4, 5]
		)

		ff = LinearModel(
			block_size=FF_LINEAR_BLOCK_SIZE,
			vocab_size=FF_LINEAR_OUTPUT_SIZE,
			dropout_rate=DROPOUT_RATE,
			layer_sizes=FF_LINEAR_LAYERS,
			hidden_activation=FF_LINEAR_ACTIVATION,
			init_fn=FF_LINEAR_INIT,
			norm=FF_LINEAR_NORM
		)

		model = Transformer(
			decoder=Decoder(
				embedding=CNNBlock(
					conv_channels=CHANNELS,
					kernel_sizes=KERNEL_SIZES,
					hidden_activation=ACTIVATION,
					pool_sizes=POOL_SIZES,
					dropout_rate=DROPOUT_RATE,
					padding=PADDING,
					avg_pool=AVG_POOL,
					norm=NORM,
					indicators=INDICATOR,
				),
				input_size=BLOCK_SIZE - EXTRA_LEN,
				emb_size=CHANNELS[-1],
				ff_size=256,
				num_heads=2
			),
			collapse=CollapseFFBlock(
				extra_len=EXTRA_LEN,
				input_channels=CHANNELS[-1],
				num_classes=VOCAB_SIZE + 1,
				linear_collapse=True,
				ff_linear=ff
			)
		)

		NP_DTYPE = np.float32
		X = np.load(
			"/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/notebook_outputs/drmca-datapreparer-copy/out/train/X/1712734175.835725.npy").astype(
			NP_DTYPE
		)

		pre_save = model(
			torch.from_numpy(X)
		).detach().numpy()

		ModelHandler.save(model, SAVE_PATH)
		post_model = ModelHandler.load(SAVE_PATH)

		post_save = post_model(
			torch.from_numpy(X)
		).detach().numpy()

		self.assertTrue(np.all(post_save == pre_save))
		self.assertIsNotNone(post_save)
