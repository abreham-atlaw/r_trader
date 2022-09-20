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
	rsi: List[int]
	wpr: List[int]
	mas_windows: List[int]
	msd_windows: List[int]
	trend_lines: List[int]
	dense_activation: Callable
	conv_activation: Callable
	loss: Callable
	optimizer: keras.optimizers.Optimizer

	def validate(self) -> bool:

		conv_out_size = self.seq_len - (max(self.mas_windows) - 1)
		for conv in self.ff_conv_pool_layers:
			conv_out_size -= (conv.size - 1)
			if conv_out_size <= 0:
				break
			if conv.pool != 0:
				conv_out_size = math.ceil(conv_out_size/conv.pool) - 1

		if conv_out_size <= 0:
			return False

		if True in [self.seq_len <= trend_line_size for trend_line_size in self.trend_lines]:
			return False

		if True in [units <= 0 for units in self.ff_dense_layers]:
			return False

		if True in [layer.features <= 0 or layer.size <= 0 for layer in self.ff_conv_pool_layers]:
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

	@staticmethod
	def __get_random_neighbor(x: int) -> int:
		return round((random.random() + 0.5)*x)

	@staticmethod
	def __get_random_mean(x0, x1, expand_bounds: bool = True, non_negative=True) -> int:
		if expand_bounds:
			x0, x1 = (5*x0 - x1)/4, (5*x1 - x0)/4
		if non_negative:
			x0, x1 = max(x0, 0), max(x1, 0)
		w = random.random()
		return round((x0*w) + ((1-w)*x1))

	@staticmethod
	def __mutate_gene(gene: Any) -> Any:

		if isinstance(gene, List):
			if len(gene) == 0:
				return  # TODO
			for _ in range(random.randint(0, len(gene))):
				index = random.randint(0, len(gene) - 1)
				gene[index] = NNConfig.__mutate_gene(gene[index])
			return gene

		if isinstance(gene, int):
			return NNConfig.__get_random_neighbor(gene)

		if isinstance(gene, bool):
			return random.choice([True, False])

		if isinstance(gene, tuple(NNConfig.CUSTOM_GENE_CLASSES)):

			for _ in range(random.randint(1, len(gene.__dict__))):
				key = random.choice(list(gene.__dict__.keys()))
				gene.__dict__[key] = NNConfig.__mutate_gene(gene.__dict__[key])
			return gene

		return gene  # TODO: CREATE A LIST OF EQUIVALENT GENES

	@staticmethod
	def __select_gene(self_value, spouse_value):

		if isinstance(self_value, List):
			swap_size = min(len(self_value), len(spouse_value))
			new_value = [NNConfig.__select_gene(self_value[i], spouse_value[i]) for i in range(swap_size)]
			length = NNConfig.__get_random_mean(len(self_value), len(spouse_value))
			if length > len(new_value) and max(len(self_value), len(spouse_value)) != 0:
				larger_genes = self_value
				if len(spouse_value) > len(self_value):
					larger_genes = spouse_value
				if len(larger_genes) < length:
					larger_genes.extend([
						NNConfig.__select_gene(
							random.choice(self_value),
							random.choice(spouse_value)
						)
						if swap_size != 0
						else random.choice(larger_genes)
						for i in range(length - len(larger_genes))
					])
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
		self.core_config, self.delta_config = self.__mutate_gene(self.core_config), self.__mutate_gene(self.delta_config)
		self.core_config.seq_len = self.delta_config.seq_len = random.choice([self.core_config.seq_len, self.delta_config.seq_len])

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
