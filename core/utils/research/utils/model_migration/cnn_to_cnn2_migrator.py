import re
import typing

from torch import nn as nn

from .model_migrator import ModelMigrator
from core.utils.research.model.model.cnn.cnn2 import CNN2
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.cnn.model import CNN


class CNNToCNN2Migrator(ModelMigrator):

	def _create_model(self, og_model: CNN) -> CNN2:
		return CNN2(
			extra_len=og_model.extra_len,
			input_size=og_model.input_size,

			embedding_block=EmbeddingBlock(
				indicators=og_model.indicators,
				positional_encoding=not isinstance(og_model.pos, nn.Identity),
				norm_positional_encoding=not isinstance(og_model.pos, nn.Identity),
			),

			cnn_block=CNNBlock(
				input_channels=og_model.layers[0].in_channels,
				conv_channels=[layer.out_channels for layer in og_model.layers],
				kernel_sizes=[layer.kernel_size for layer in og_model.layers],
				pool_sizes=[
					0 if isinstance(layer, nn.Identity)
					else (layer.pool_range[0], layer.pool_range[1], layer.pool.kernel_size[0])
					for layer in og_model.pool_layers
				],
				hidden_activation=[
					nn.Identity() if isinstance(layer, nn.Identity)
					else layer
					for layer in og_model.hidden_activations
				],
				dropout_rate=[
					0 if isinstance(layer, nn.Identity)
					else layer.p
					for layer in og_model.dropouts[:-1]
				],
				norm=[
					not isinstance(layer, nn.Identity)
					for layer in og_model.norm_layers
				],
				stride=[
					layer.stride
					for layer in og_model.layers
				],
				padding=og_model.layers[0].padding
			),

			collapse_block=CollapseBlock(
				dropout=0 if isinstance(og_model.dropouts[-1], nn.Identity) else og_model.dropouts[-1].p,
				ff_block=og_model.ff_block
			)
		)

	def _get_pattern_generator_mapping(self) -> typing.Dict[str, typing.Callable]:
		return {
			r"^layers\.\d+\.weight$":
				lambda key: "cnn_block.layers.{}.weight".format(re.match(r"^layers\.(\d+)\.weight$", key).group(1)),

			r"^layers\.\d+\.bias":
				lambda key: "cnn_block.layers.{}.bias".format(re.match(r"^layers\.(\d+)\.bias$", key).group(1)),

			r"^ff_block.layers\.\d+\.weight":
				lambda key: "collapse_block.ff_block.layers.{}.weight".format(
					re.match(r"^ff_block\.layers\.(\d+)\.weight$", key).group(1)
				),

			r"^ff_block.layers\.\d+\.bias":
				lambda key: "collapse_block.ff_block.layers.{}.bias".format(
					re.match(r"^ff_block\.layers\.(\d+)\.bias$", key).group(1)
				),

			r"^norm_layers\.\d+\.norm_layer.weight":
				lambda key: "cnn_block.norm_layers.{}.norm_layer.weight".format(
					re.match(r"^norm_layers\.(\d+)\.norm_layer\.weight$", key).group(1)
				),

			r"^norm_layers\.\d+\.norm_layer.bias":
				lambda key: "cnn_block.norm_layers.{}.norm_layer.bias".format(
					re.match(r"^norm_layers\.(\d+)\.norm_layer\.bias$", key).group(1)
				)
		}
