from typing import *

import math
import random
from dataclasses import dataclass

from tensorflow import keras

from lib.ga import Species


@dataclass
class ConvPoolLayer:
	size: int
	features: int
	pool: int


@dataclass
class ModelConfig:

	seq_len: int
	ff_dense_layers: List[int]
	ff_conv_pool_layers: List[ConvPoolLayer]
	delta: bool
	norm: bool
	stochastic_oscillators: List[int]
	trend_lines: List[int]
	mas_windows: List[int]
	dense_activation: Callable
	conv_activation: Callable
	loss: Callable
	optimizer: keras.optimizers.Optimizer

	def validate(self) -> bool:

		conv_out_size = self.seq_len - (max(self.mas_windows) - 1)
		for conv in self.ff_conv_pool_layers:
			conv_out_size -= (conv.size - 1)
			if conv.pool != 0:
				conv_out_size = math.ceil(conv_out_size/conv.pool) - 1

		if conv_out_size <= 0:
			return False

		if True in [self.seq_len <= trend_line_size for trend_line_size in self.trend_lines]:
			return False

		return True


class NNConfig(Species):

	CUSTOM_GENE_CLASSES = [
		ModelConfig,
		ConvPoolLayer
	]

	def __init__(self, core_config: ModelConfig, delta_config: ModelConfig):
		self.core_config = core_config
		self.delta_config = delta_config

	def __mutate_seq_len(self):
		change_range = math.ceil(self.core_config.seq_len / 10)
		seq_len = self.core_config.seq_len + random.randint(-change_range, change_range)
		self.core_config.seq_len = self.delta_config.seq_len = seq_len

	@staticmethod
	def __mutate_individual_ff_dense_layers(layers: List[int]):
		index = random.randint(0, len(layers)-1)
		layers[index] *= math.ceil(random.randint(5, 15) / 10)

	@staticmethod
	def __mutate__individual_ff_conv_pool_layers(layers: List[ConvPoolLayer]):
		index = random.randint(0, len(layers)-1)
		layer = layers[index]
		layer.size *= math.ceil(random.randint(5, 15) / 10)

		pool_change_range = math.ceil(layer.pool / 5)
		layer.pool += random.randint(-pool_change_range, pool_change_range)

		layer.features *= math.ceil(random.randint(5, 15) / 10)

	def __mutate_ff_dense_layers(self):
		self.__mutate_individual_ff_dense_layers(
			random.choice([self.core_config, self.delta_config]).ff_dense_layers
		)

	def __mutate_ff_conv_layers(self):
		self.__mutate__individual_ff_conv_pool_layers(
			random.choice([self.core_config, self.delta_config]).ff_conv_pool_layers
		)

	@staticmethod
	def __get_random_mean(x0, x1) -> int:
		w = random.random()
		return round((x0*w) + ((1-w)*x1))

	@staticmethod
	def __select_gene(self_value, spouse_value):

		if isinstance(self_value, List):
			swap_size = min(len(self_value), len(spouse_value))
			new_value = [NNConfig.__select_gene(self_value[i], spouse_value[i]) for i in range(swap_size)]
			length = NNConfig.__get_random_mean(len(self_value), len(spouse_value))
			if length > len(new_value):
				larger_genes = self_value
				if len(spouse_value) > len(self_value):
					larger_genes = spouse_value
				new_value.extend(larger_genes[len(new_value): length])
			return new_value

		if isinstance(self_value, int) and not isinstance(self_value, bool):
			return NNConfig.__get_random_mean(self_value, spouse_value)

		if isinstance(self_value, tuple(NNConfig.CUSTOM_GENE_CLASSES)):
			return self_value.__class__(**{
				key: NNConfig.__select_gene(self_value.__dict__[key], spouse_value.__dict__[key])
				for key in self_value.__dict__.keys()
			})

		return random.choice((self_value, spouse_value))

	@staticmethod
	def __generate_offspring_model_config(
			self_config: ModelConfig,
			spouse_config: ModelConfig,
			seq_len: int
	) -> ModelConfig:

		config: ModelConfig = NNConfig.__select_gene(self_config, spouse_config)
		config.seq_len = seq_len

		return config

	def __generate_offspring(self, spouse: 'NNConfig') -> 'NNConfig':
		seq_len = random.choice((spouse.core_config.seq_len, self.core_config.seq_len))
		return NNConfig(
			self.__generate_offspring_model_config(
				self.core_config,
				spouse.core_config,
				seq_len
			),
			self.__generate_offspring_model_config(
				self.delta_config,
				spouse.core_config,
				seq_len
			)
		)

	def mutate(self, *args, **kwargs):
		random.choice(
			[self.__mutate_seq_len, self.__mutate_ff_conv_layers, self.__mutate_ff_dense_layers]
		)()

	def reproduce(self, spouse: 'NNConfig', preferred_offsprings: int) -> List['NNConfig']:
		offsprings = []
		while len(offsprings) < preferred_offsprings:
			offspring = self.__generate_offspring(spouse)
			if offspring.validate():
				offsprings.append(offspring)
		return offsprings

	def validate(self) -> bool:
		return self.core_config.validate() and self.delta_config.validate()

	def __str__(self):
		return f"\nCore Config: {str(self.core_config)}\nDeltaConfig: {str(self.delta_config)}\n"
