import unittest

from torch import nn

from core import Config
from core.utils.ganno.torch.builder import ModelBuilder
from core.utils.ganno.torch.nnconfig import CNNConfig, ConvLayer, TransformerConfig, LinearConfig
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Transformer


class BuilderTest(unittest.TestCase):

	def test_cnn_build(self):

		builder = ModelBuilder()
		config = CNNConfig(
			layers=[
				ConvLayer(
					kernel_size=kernel_size,
					features=features,
					pooling=pooling,
				)
				for kernel_size, features, pooling in [
					(3, 32, 2),
					(5, 16, 3)
				]
			],
			dropout=0.1,
			vocab_size=Config.VOCAB_SIZE,
			ff_block=LinearConfig(
				vocab_size=Config.VOCAB_SIZE,
				layers=[128, 128, 432],
				dropout=0.5,

			)
		)

		model = builder.build(config)

		self.assertIsInstance(model, CNN)

	def test_transformer_build(self):
		builder = ModelBuilder()
		config = TransformerConfig(
			kernel_size=3,
			num_heads=4,
			emb_size=8,
			block_size=1024,
			ff_size=128,
			vocab_size=499
		)

		model = builder.build(config)

		self.assertIsInstance(model, Transformer)

	def test_linear_build(self):

		builder = ModelBuilder()
		config = LinearConfig(
			449,
			layers=[256, 512],
			block_size=1024,
			dropout=0.1
		)

		model = builder.build(config)

		self.assertIsInstance(model, LinearModel)
