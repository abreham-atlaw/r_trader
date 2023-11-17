from torch import nn

from core.utils.ganno.torch.nnconfig import ModelConfig, CNNConfig, TransformerConfig
from core.utils.research.model.model.cnn.model import CNN
from core.utils.research.model.model.transformer import Transformer, Decoder


class ModelBuilder:

	def __build_conv(self, config: CNNConfig) -> nn.Module:
		return CNN(
			num_classes=config.vocab_size,
			conv_channels=[1]+[layer.features for layer in config.layers],
			kernel_sizes=[layer.kernel_size for layer in config.layers],
			pool_sizes=[layer.pooling for layer in config.layers],
			dropout_rate=config.dropout
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

	def build(self, config: ModelConfig) -> nn.Module:

		if isinstance(config, CNNConfig):
			model = self.__build_conv(config)
		elif isinstance(config, TransformerConfig):
			model = self.__build_transformer(config)
		else:
			raise Exception(f"Unknown Config Type {config.__class__.__name__}")

		return model
