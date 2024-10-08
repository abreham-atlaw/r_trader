from torch import nn

from core.Config import FF_LINEAR_BLOCK_SIZE, FF_LINEAR_OUTPUT_SIZE, FF_LINEAR_LAYERS, FF_LINEAR_INIT, FF_LINEAR_NORM, \
	INDICATORS_DELTA, INDICATORS_SO, INDICATORS_RSI
from core.utils.ganno.torch.nnconfig import ModelConfig, CNNConfig, TransformerConfig, LinearConfig
from core.utils.research.model.layers import Indicators
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Transformer, Decoder


class ModelBuilder:

	def __build_conv(self, config: CNNConfig) -> nn.Module:
		return CNN(
			extra_len=config.extra_len,
			num_classes=config.vocab_size + 1,
			conv_channels=[layer.features for layer in config.layers],
			kernel_sizes=[layer.kernel_size for layer in config.layers],
			hidden_activation=nn.ReLU(),
			pool_sizes=[layer.pooling for layer in config.layers],
			dropout_rate=config.dropout,
			padding=0,
			linear_collapse=True,
			norm=[layer.norm for layer in config.layers],
			ff_linear=LinearModel(
				block_size=FF_LINEAR_BLOCK_SIZE,
				vocab_size=FF_LINEAR_OUTPUT_SIZE,
				dropout_rate=config.dropout,
				layer_sizes=FF_LINEAR_LAYERS,
				hidden_activation=nn.ReLU(),
				init_fn=FF_LINEAR_INIT,
				norm=FF_LINEAR_NORM
			),
			indicators=Indicators(
				delta=INDICATORS_DELTA,
				so=INDICATORS_SO,
				rsi=INDICATORS_RSI
			),
			input_size=config.block_size
		)

	def __build_transformer(self, config: TransformerConfig) -> nn.Module:
		return Transformer(
			Decoder(
				kernel_size=config.kernel_size,
				emb_size=config.emb_size,
				block_size=config.block_size,
				num_heads=config.num_heads,
				ff_size=config.ff_size
			),
			vocab_size=config.vocab_size
		)

	def __build_linear(self, config: LinearConfig) -> nn.Module:
		return LinearModel(
			block_size=config.block_size,
			vocab_size=config.vocab_size,
			layer_sizes=config.layers,
			dropout_rate=config.dropout,
			hidden_activation=nn.ReLU()
		)

	def build(self, config: ModelConfig) -> nn.Module:

		if isinstance(config, CNNConfig):
			model = self.__build_conv(config)
		elif isinstance(config, TransformerConfig):
			model = self.__build_transformer(config)
		elif isinstance(config, LinearConfig):
			model = self.__build_linear(config)
		else:
			raise Exception(f"Unknown Config Type {config.__class__.__name__}")

		return model
