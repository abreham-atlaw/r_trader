import typing
from dataclasses import dataclass
from enum import Enum
import random

import torch.nn as nn
from torch.optim import Optimizer

from core.Config import ModelConfig
from lib.ga import Species


@dataclass
class ConvLayer:
	kernel_size: int
	features: int
	padding: int = 1
	pooling: int = 0


@dataclass
class ModelConfig(Species):
	vocab_size: int

	def mutate(self, *args, **kwargs):
		pass


@dataclass
class CNNConfig(ModelConfig):
	layers: typing.List[ConvLayer]
	dropout: float = 0

	def reproduce(self, spouse: 'CNNConfig', preferred_offsprings: int) -> typing.List['Species']:
		configs = []
		for _ in range(preferred_offsprings):
			layers1 = self.layers
			layers2 = spouse.layers
			mixed_layers = []
			for l1, l2 in zip(layers1, layers2):
				mixed_layers.append(random.choice([l1, l2]))

			dropout = random.choice([self.dropout, spouse.dropout])
			vocab_size = random.choice([self.vocab_size, spouse.vocab_size])
			new_config = CNNConfig(layers=mixed_layers, dropout=dropout, vocab_size=vocab_size)
			configs.append(new_config)
		return configs


@dataclass
class TransformerConfig(ModelConfig):
	kernel_size: int
	emb_size: int
	block_size: int
	num_heads: int
	ff_size: int

	def reproduce(self, spouse: 'TransformerConfig', preferred_offsprings: int) -> typing.List['Species']:
		configs = []
		for _ in range(preferred_offsprings):
			kernel_size = random.choice([self.kernel_size, spouse.kernel_size])
			emb_size = random.choice([self.emb_size, spouse.emb_size])
			block_size = random.choice([self.block_size, spouse.block_size])
			num_heads = random.choice([self.num_heads, spouse.num_heads])
			ff_size = random.choice([self.ff_size, spouse.ff_size])
			vocab_size = random.choice([self.vocab_size, spouse.vocab_size])
			new_config = TransformerConfig(
				kernel_size=kernel_size,
				emb_size=emb_size,
				block_size=block_size,
				num_heads=num_heads,
				ff_size=ff_size,
				vocab_size=vocab_size
			)
			configs.append(new_config)
		return configs
