from typing import *

from tensorflow import keras

import random
import math
from dataclasses import dataclass

from lib.ga import GeneticAlgorithm
from lib.utils.logger import Logger
from .builder import ModelBuilder, CoreBuilder, DeltaBuilder
from .trainer import Trainer
from .nnconfig import NNConfig, ConvPoolLayer, ModelConfig


@dataclass
class ListRangeConfig:
	length_range: Tuple[int, int]
	values_range: Tuple[int, int]


@dataclass
class NNInitialPopulationConfig:

	size: int = 10

	seq_len_range: Tuple[int, int] = (16, 256)
	dnn_layer_range: ListRangeConfig = ListRangeConfig(
		length_range=(3, 6),
		values_range=(16, 2048)
	)

	conv_layer_depth_range: Tuple[int, int] = (2, 4)
	conv_layer_features_range: Tuple[int, int] = (16, 256)
	conv_layer_size_range: Tuple[int, int] = (2, 16)
	conv_layer_pooling_range: Tuple[int, int] = (0, 4)

	stochastic_oscillators_range: ListRangeConfig = ListRangeConfig(
		length_range=(1, 16),
		values_range=(5, 128)
	)
	trend_lines_range: ListRangeConfig = ListRangeConfig(
		length_range=(1, 4),
		values_range=(5, 64)
	)
	moving_averages_range: ListRangeConfig = ListRangeConfig(
		length_range=(1, 10),
		values_range=(5, 64)
	)

	dense_activations: List[Callable] = (
		keras.activations.relu,
		keras.activations.tanh,
		keras.activations.sigmoid,
		keras.activations.softplus,
		keras.activations.softsign
	)
	conv_activations: List[Callable] = (
		keras.activations.relu,
		keras.activations.tanh,
		keras.activations.sigmoid,
		keras.activations.softplus,
		keras.activations.softsign
	)
	core_loss_functions: List[Callable] = (keras.losses.binary_crossentropy,)
	delta_loss_functions: List[Callable] = (
		keras.losses.mean_squared_error,
		keras.losses.mean_absolute_error,
		keras.losses.mean_squared_logarithmic_error,
		keras.losses.mean_absolute_percentage_error
	)
	optimizers: List[Type] = (
		keras.optimizers.Adam(),
		keras.optimizers.SGD()
	)


class NNGeneticAlgorithm(GeneticAlgorithm):

	def __init__(
			self,
			trainer: Trainer,
			*args,
			core_builder: Optional[ModelBuilder] = None,
			delta_builder: Optional[ModelBuilder] = None,
			initial_population_config: Optional[NNInitialPopulationConfig] = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__trainer = trainer
		self.__initial_population_config = initial_population_config
		self.__core_builder = core_builder
		if core_builder is None:
			self.__core_builder = CoreBuilder()
		self.__delta_builder = delta_builder
		if delta_builder is None:
			self.__delta_builder = DeltaBuilder()
		if initial_population_config is None:
			self.__initial_population_config = NNInitialPopulationConfig()

	@staticmethod
	def __generate_random_int_list(range_config: ListRangeConfig, value_upper_bound: Optional[int] = None) -> List[int]:
		if value_upper_bound is not None:
			range_config = ListRangeConfig(
				length_range=range_config.length_range,
				values_range=(
					range_config.values_range[0],
					min(
						range_config.values_range[1],
						value_upper_bound
					)
				)
			)
		return [random.randint(*range_config.values_range) for _ in range(random.randint(*range_config.length_range))]

	@staticmethod
	def __generate_random_cnn_layers(input_size, depth_range, features_range, size_range, pooling_range) -> List[ConvPoolLayer]:

		remaining_size = input_size
		layers = []
		for _ in range(random.randint(*depth_range)):
			if remaining_size <= size_range[0]+1:
				break
			size = random.randint(size_range[0], min(remaining_size, size_range[1]))
			remaining_size -= (size - 1)
			pool = random.randint(pooling_range[0], min(remaining_size-1, pooling_range[1]))
			if pool != 0:
				remaining_size = math.ceil(remaining_size/pool) - 1
			layers.append(
				ConvPoolLayer(
					size=size,
					features=random.randint(*features_range),
					pool=pool
				)
			)

		return layers

	def __generate_random_model(self, seq_len, loss) -> ModelConfig:
		mas_windows = self.__generate_random_int_list(self.__initial_population_config.moving_averages_range, value_upper_bound=seq_len-1)

		return ModelConfig(
			seq_len=seq_len,
			ff_dense_layers=self.__generate_random_int_list(self.__initial_population_config.dnn_layer_range),
			ff_conv_pool_layers=self.__generate_random_cnn_layers(
				input_size=seq_len - (max(mas_windows)-1),
				depth_range=self.__initial_population_config.conv_layer_depth_range,
				size_range=self.__initial_population_config.conv_layer_size_range,
				features_range=self.__initial_population_config.conv_layer_features_range,
				pooling_range=self.__initial_population_config.conv_layer_pooling_range
			),
			stochastic_oscillators=self.__generate_random_int_list(self.__initial_population_config.stochastic_oscillators_range),
			trend_lines=self.__generate_random_int_list(
				self.__initial_population_config.trend_lines_range,
				value_upper_bound=seq_len-1
			),
			mas_windows=mas_windows,
			delta=random.choice((True, False)),
			norm=random.choice((True, False)),
			dense_activation=random.choice(self.__initial_population_config.dense_activations),
			conv_activation=random.choice(self.__initial_population_config.conv_activations),
			loss=loss,
			optimizer=random.choice(self.__initial_population_config.optimizers)
		)

	def __generate_random_core_model(self, seq_len) -> ModelConfig:
		return self.__generate_random_model(seq_len, random.choice(self.__initial_population_config.core_loss_functions))

	def __generate_random_delta_model(self, seq_len) -> ModelConfig:
		return self.__generate_random_model(seq_len, random.choice(self.__initial_population_config.delta_loss_functions))

	def __generate_random_config(self) -> NNConfig:
		seq_len = random.randint(*self.__initial_population_config.seq_len_range)
		return NNConfig(
			core_config=self.__generate_random_core_model(seq_len),
			delta_config=self.__generate_random_delta_model(seq_len)
		)

	def __build_models(self, config: NNConfig) -> Tuple[keras.models.Model, keras.models.Model]:
		return self.__core_builder.build(config.core_config), self.__delta_builder.build(config.delta_config)

	@Logger.logged_method
	def _generate_initial_generation(self) -> List[NNConfig]:
		Logger.info("Generating Initial Generation.")
		population = []
		while len(population) < self.__initial_population_config.size:
			config = self.__generate_random_config()
			if config.validate():
				population.append(config)
		return population

	@Logger.logged_method
	def _evaluate_species(self, species: NNConfig) -> float:
		Logger.info("Building Models")
		core_model, delta_model = self.__build_models(species)
		Logger.info("Fitting Models")
		self.__trainer.fit(core_model, delta_model)
		Logger.info("Evaluating Models")
		return 1/self.__trainer.evaluate(core_model, delta_model)
