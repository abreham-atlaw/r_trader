import typing
from dataclasses import dataclass
from enum import Enum
import random

import torch.nn as nn
from torch.optim import Optimizer

from core.Config import ModelConfig
from core.utils.ganno.torch.choice_utils import ChoiceUtils
from lib.ga import Species


@dataclass
class ConvLayer:
	kernel_size: int
	features: int
	padding: int = 1
	pooling: int = 0
	norm: bool = False


@dataclass
class ModelConfig(Species):
	vocab_size: int

	def mutate(self, *args, **kwargs):
		pass


@dataclass
class LinearConfig(ModelConfig):
	layers: typing.List[int]
	dropout: float
	norm: typing.List[bool]

	def generate_offspring(self: 'LinearConfig', spouse: 'LinearConfig') -> 'LinearConfig':
		min_len = min(len(self.layers), len(spouse.layers))
		max_len = max(len(self.layers), len(spouse.layers))
		new_len = random.randint(min_len, max_len)

		layers_pool = self.layers + spouse.layers
		random.shuffle(layers_pool)

		new_layers = layers_pool[:new_len]

		return LinearConfig(
			vocab_size=self.vocab_size,

			layers=ChoiceUtils.list_select(
				self.layers,
				spouse.layers,
				discrete=False,
				round_mode=True
			),

			dropout=ChoiceUtils.choice_continuous(
				self.dropout,
				spouse.dropout,
				noise=0.1,
				min_value=0,
				max_value=1
			),

			norm=ChoiceUtils.list_select(
				self.norm,
				spouse.norm,
				discrete=True
			)
		)

	def reproduce(self, spouse: 'LinearConfig', preferred_offsprings: int) -> typing.List['LinearConfig']:
		return [self.generate_offspring(spouse) for _ in range(preferred_offsprings)]


@dataclass
class CNNConfig(ModelConfig):
	layers: typing.List[ConvLayer]
	ff_block: LinearConfig
	dropout: float = 0
	extra_len: int = 124
	block_size: int = 1024 + extra_len

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
			new_config = CNNConfig(
				layers=mixed_layers,
				dropout=dropout,
				vocab_size=vocab_size,
				ff_block=self.ff_block.generate_offspring(
					spouse=spouse.ff_block,
				),

			)
			configs.append(new_config)
		return configs


@dataclass
class TransformerConfig(ModelConfig):
	kernel_size: int
	emb_size: int
	num_heads: int
	ff_size: int

	def reproduce(self, spouse: 'TransformerConfig', preferred_offsprings: int) -> typing.List['Species']:
		configs = []
		for _ in range(preferred_offsprings):
			kernel_size = random.choice([self.kernel_size, spouse.kernel_size])
			emb_size = random.choice([self.emb_size, spouse.emb_size])
			num_heads = random.choice([self.num_heads, spouse.num_heads])
			ff_size = random.choice([self.ff_size, spouse.ff_size])
			vocab_size = random.choice([self.vocab_size, spouse.vocab_size])
			new_config = TransformerConfig(
				kernel_size=kernel_size,
				emb_size=emb_size,
				num_heads=num_heads,
				ff_size=ff_size,
				vocab_size=vocab_size
			)
			configs.append(new_config)
		return configs


