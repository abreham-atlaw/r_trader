from typing import *

import math
import random
from dataclasses import dataclass

from tensorflow import keras

from lib.ga.species import ClassDictSpecies


@dataclass
class ConvPoolLayer:
	size: int
	features: int
	pool: int
	dropout: float


@dataclass
class ModelConfig:

	seq_len: int
	ff_dense_layers: List[Tuple[int, float]]
	ff_conv_pool_layers: List[ConvPoolLayer]
	delta: bool
	norm: bool
	include_prep: bool
	stochastic_oscillators: List[int]
	rsi: List[int]
	wpr: List[int]
	kelman_filters: int
	kelman_static_filters: List[Tuple[float, float]]
	mas_windows: List[int]
	msd_windows: List[int]
	trend_lines: List[int]
	dense_activation: Callable
	conv_activation: Callable
	loss: Callable
	optimizer: keras.optimizers.Optimizer

	def validate(self) -> bool:

		conv_out_size = self.seq_len - (max([
			max(oi)
			for oi in [
				self.stochastic_oscillators,
				self.rsi,
				self.wpr,
				self.mas_windows,
				self.msd_windows,
			]]) - 1)
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


class NNConfig(ClassDictSpecies):

	CUSTOM_GENE_CLASSES = [
		ModelConfig,
		ConvPoolLayer
	]

	def __init__(self, core_config: ModelConfig, delta_config: ModelConfig):
		super().__init__()
		self.core_config = core_config
		self.delta_config = delta_config

	def _get_protected_list_genes(self) -> List[str]:
		return ['ff_conv_pool_layers']

	def _get_gene_classes(self):
		return NNConfig.CUSTOM_GENE_CLASSES

	def mutate(self, *args, **kwargs):
		super().mutate(*args, **kwargs)
		self.core_config.seq_len = self.delta_config.seq_len = random.choice([self.core_config.seq_len, self.delta_config.seq_len])

	def _generate_offspring(self, spouse) -> Optional['NNConfig']:
		try:
			offspring: NNConfig = super()._generate_offspring(spouse)
			offspring.core_config.seq_len = offspring.delta_config.seq_len = max(offspring.core_config.seq_len, offspring.delta_config.seq_len)
			if offspring.validate():
				return offspring
			else:
				del offspring
				return self._generate_offspring(spouse)
		except RecursionError:
			return None

	def reproduce(self, spouse: 'Species', preferred_offsprings: int) -> List['Species']:
		return [offspring for offspring in super().reproduce(spouse, preferred_offsprings) if offspring is not None]

	def validate(self) -> bool:
		return self.core_config.validate() and self.delta_config.validate()

	def __str__(self):
		return f"\nCore Config: {str(self.core_config)}\nDeltaConfig: {str(self.delta_config)}\n"
