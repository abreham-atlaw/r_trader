from typing import *

import math
import random
from dataclasses import dataclass

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
	dense_activation: Callable
	conv_activation: Callable
	loss: Callable
	optimizer: Type

	def validate(self) -> bool:

		conv_out_size = self.seq_len
		for conv in self.ff_conv_pool_layers:
			conv_out_size -= (conv.size - 1)
			if conv.pool != 0:
				conv_out_size = math.ceil(conv_out_size/conv.pool) - 1

		if conv_out_size <= 0:
			return False

		return True


class NNConfig(Species):

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
	def __generate_offspring_model_config(
			self_config: ModelConfig,
			spouse_config: ModelConfig,
			seq_len: int
	) -> ModelConfig:

		choices = [
			random.choice(choice)
			for choice in [
				(self_config.ff_dense_layers, spouse_config.ff_dense_layers),
				(self_config.ff_conv_pool_layers, spouse_config.ff_conv_pool_layers),
				(self_config.delta, spouse_config.delta),
				(self_config.norm, spouse_config.norm),
				(self_config.dense_activation, spouse_config.dense_activation),
				(self_config.conv_activation, spouse_config.conv_activation),
				(self_config.loss, spouse_config.loss),
				(self_config.optimizer, spouse_config.optimizer)
			]
		]
		return ModelConfig(seq_len, *choices)

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
