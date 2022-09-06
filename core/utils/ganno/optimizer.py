from typing import *

from tensorflow import keras

import random
from dataclasses import dataclass

from lib.ga import GeneticAlgorithm
from lib.utils.logger import Logger
from .builder import ModelBuilder, CoreBuilder, DeltaBuilder
from .trainer import Trainer
from .nnconfig import NNConfig, ConvPoolLayer, ModelConfig


@dataclass
class NNInitialPopulationConfig:

	size: int = 10

	seq_len_range: Tuple[int, int] = (16, 256)
	dnn_layer_depth_range: Tuple[int, int] = (3, 6)
	dnn_layer_breadth_range: Tuple[int, int] = (16, 2048)
	conv_layer_depth_range: Tuple[int, int] = (2, 6)
	conv_layer_features_range: Tuple[int, int] = (16, 256)
	conv_layer_size_range: Tuple[int, int] = (2, 64)
	conv_layer_pooling_range: Tuple[int, int] = (0, 16)
	stochastic_oscillator_size_range: Tuple[int, int] = (5, 128)
	trend_line_size_range: Tuple[int, int] = (5, 256)
	mas_amounts_range: Tuple[int, int] = (1, 10)
	mas_windows_range: Tuple[int, int] = (5, 64)
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
	def __generate_random_dnn_layers(depth_range: Tuple[int, int], breadth_range: Tuple[int, int]) -> List[int]:
		return [random.randint(*breadth_range) for _ in range(random.randint(*depth_range))]

	@staticmethod
	def __generate_random_cnn_layers(depth_range, features_range, size_range, pooling_range) -> List[ConvPoolLayer]:
		return [
			ConvPoolLayer(
				size=random.randint(*size_range),
				features=random.randint(*features_range),
				pool=random.randint(*pooling_range)
			)
			for _ in range(random.randint(*depth_range))
		]

	@staticmethod
	def __generate_random_mas(amounts_range: Tuple[int, int], windows_range: Tuple[int, int]) -> List[int]:
		return [
			random.randint(*windows_range)
			for _ in range(random.randint(*amounts_range))
		]

	def __generate_random_model(self, seq_len, loss) -> ModelConfig:
		return ModelConfig(
			seq_len=seq_len,
			ff_dense_layers=self.__generate_random_dnn_layers(
				depth_range=self.__initial_population_config.dnn_layer_depth_range,
				breadth_range=self.__initial_population_config.dnn_layer_breadth_range
			),
			ff_conv_pool_layers=self.__generate_random_cnn_layers(
				depth_range=self.__initial_population_config.conv_layer_depth_range,
				size_range=self.__initial_population_config.conv_layer_size_range,
				features_range=self.__initial_population_config.conv_layer_features_range,
				pooling_range=self.__initial_population_config.conv_layer_pooling_range
			),
			stochastic_oscillator_size=random.randint(*self.__initial_population_config.stochastic_oscillator_size_range),
			trend_line_size=random.randint(*self.__initial_population_config.trend_line_size_range),
			mas_windows=self.__generate_random_mas(self.__initial_population_config.mas_amounts_range, self.__initial_population_config.mas_windows_range),
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
