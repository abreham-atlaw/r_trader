import os

from torch import nn

from core import Config
from core.utils.research.losses import MeanSquaredErrorLoss
from core.utils.research.model.layers import DynamicLayerNorm, DynamicBatchNorm, Indicators
from core.utils.research.model.model.cnn.cnn2 import CNN2
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.linear.model import LinearModel
from .trainer_test import TrainerTest


class LassTrainerTest(TrainerTest):

	def _get_root_dirs(self):
		return [
			os.path.join(Config.BASE_DIR, "temp/Data/lass/1/train")
		], [
			os.path.join(Config.BASE_DIR, "temp/Data/lass/1/test")
		]

	def _create_losses(self):
		return (
			None,
			MeanSquaredErrorLoss(weighted_sample=False)
		)

	def _create_model(self):
		INPUT_CHANNELS = 2

		EMBEDDING_SIZE = 32

		CHANNELS = [EMBEDDING_SIZE for _ in range(4)]
		EXTRA_LEN = self._get_extra_len()
		KERNEL_SIZES = [3 for _ in CHANNELS]
		VOCAB_SIZE = self._get_vocab_size()
		POOL_SIZES = [0 for _ in CHANNELS]
		DROPOUT_RATE = [0 for _ in CHANNELS]
		ACTIVATION = [nn.Identity(), nn.Identity(), nn.LeakyReLU(), nn.Identity()]
		BLOCK_SIZE = self._get_sequence_length() + EXTRA_LEN
		PADDING = 0
		NORM = [DynamicLayerNorm()] + [nn.Identity() for _ in CHANNELS[1:]]

		INDICATORS_DELTA = []
		INDICATORS_SO = []
		INDICATORS_RSI = []
		INPUT_NORM = DynamicLayerNorm()

		COLLAPSE_INPUT_NORM = DynamicBatchNorm()
		DROPOUT_BRIDGE = 0.2
		COLLAPSE_GLOBAL_AVG_POOL = True
		COLLAPSE_EXTRA_MODE = False

		FF_LINEAR_LAYERS = [64, 16] + [VOCAB_SIZE + 1]
		FF_LINEAR_ACTIVATION = [nn.Identity(), nn.LeakyReLU()]
		FF_LINEAR_INIT = None
		FF_LINEAR_NORM = [nn.Identity() for _ in FF_LINEAR_LAYERS[:]]
		FF_DROPOUT = 0

		indicators = Indicators(
			delta=INDICATORS_DELTA,
			so=INDICATORS_SO,
			rsi=INDICATORS_RSI,
			input_channels=INPUT_CHANNELS
		)

		return CNN2(
			extra_len=EXTRA_LEN,
			input_size=(None, INPUT_CHANNELS, BLOCK_SIZE) if INPUT_CHANNELS > 1 else BLOCK_SIZE,

			embedding_block=EmbeddingBlock(
				indicators=indicators,
				input_norm=INPUT_NORM
			),

			cnn_block=CNNBlock(
				input_channels=indicators.indicators_len,
				conv_channels=CHANNELS,
				kernel_sizes=KERNEL_SIZES,
				pool_sizes=POOL_SIZES,
				hidden_activation=ACTIVATION,
				dropout_rate=DROPOUT_RATE,
				norm=NORM,
				padding=PADDING
			),

			collapse_block=CollapseBlock(
				extra_mode=COLLAPSE_EXTRA_MODE,
				dropout=DROPOUT_BRIDGE,
				input_norm=COLLAPSE_INPUT_NORM,
				global_avg_pool=COLLAPSE_GLOBAL_AVG_POOL,
				ff_block=LinearModel(
					dropout_rate=FF_DROPOUT,
					layer_sizes=FF_LINEAR_LAYERS,
					hidden_activation=FF_LINEAR_ACTIVATION,
					init_fn=FF_LINEAR_INIT,
					norm=FF_LINEAR_NORM
				)
			)

		)

	def _get_sequence_length(self):
		return 32

	def _get_extra_len(self):
		return 0

	def _get_vocab_size(self) -> int:
		return 0

	def _get_reg_loss_only(self) -> bool:
		return True

	def test_train(self):
		super().test_train()
