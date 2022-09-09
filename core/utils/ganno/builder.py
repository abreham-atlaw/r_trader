from typing import *
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, MaxPooling1D, Input, Reshape, Concatenate, Flatten
from tensorflow.keras.activations import sigmoid

from lib.utils.logger import Logger
from lib.dnn.layers import Delta, Norm, UnNorm, StochasticOscillator, MultipleMovingAverages, TrendLine
from .nnconfig import ModelConfig, ConvPoolLayer


class ModelBuilder(ABC):

	def __init__(self, un_norm=False, output_activation=None, summarize=True):
		self.__un_norm = un_norm
		self.__output_activation = output_activation
		self.__summarize = summarize

	@staticmethod
	def __concat_layers(inputs: Layer, layer_class: Type, args: List[Any], axis=1) -> Layer:
		layers = [layer_class(*arg)(inputs) for arg in args]
		return Concatenate(
			axis=axis
		)(layers)

	@staticmethod
	def _add_ff_dense_layers(layer: Layer, layers: List[int], activation: Callable) -> Layer:
		for layer_size in layers:
			layer = Dense(layer_size, activation=activation)(layer)
		return layer

	@staticmethod
	def _add_ff_conv_layers(layer: Layer, layers: List[ConvPoolLayer], activation: Callable) -> Layer:
		for config in layers:
			layer = Conv1D(kernel_size=config.size, filters=config.features, activation=activation)(layer)
			if config.pool != 0:
				layer = MaxPooling1D(pool_size=config.pool)(layer)
		return layer

	@abstractmethod
	def _get_input_shape(self, seq_len: int) -> int:
		pass

	def _compile(self, model: Model, optimizer: keras.optimizers.Optimizer, loss: Callable):
		model.compile(optimizer=optimizer, loss=loss)

	def build(self, config: ModelConfig) -> Model:
		Logger.info("[+]Building", config)
		input_layer = Input(shape=self._get_input_shape(config.seq_len))

		input_sequence = input_layer[:, :config.seq_len]
		extra_input = input_layer[:, config.seq_len:]

		prep_layer = input_sequence
		if config.delta:
			prep_layer = Delta()(prep_layer)
		if config.norm:
			prep_layer = Norm()(prep_layer)

		mas = MultipleMovingAverages(config.mas_windows)(prep_layer)

		ff_conv = self._add_ff_conv_layers(mas, config.ff_conv_pool_layers, config.conv_activation)

		sos = self.__concat_layers(
			input_sequence,
			StochasticOscillator,
			[(arg,) for arg in config.stochastic_oscillators]
		)
		trend_lines = self.__concat_layers(
			input_sequence,
			TrendLine,
			[(arg,) for arg in config.trend_lines]
		)

		flatten = Concatenate(axis=1)((
			Flatten()(ff_conv),
			sos,
			trend_lines,
			extra_input
		))

		ff_dense = self._add_ff_dense_layers(flatten, config.ff_dense_layers, config.dense_activation)

		output_layer = Dense(1, activation=self.__output_activation)(ff_dense)

		if self.__un_norm:
			output_layer = UnNorm(min_increment=False)(
				Concatenate(axis=1)((
					input_sequence,
					output_layer
				))
			)
			output_layer = Reshape((1,))(output_layer)

		model = Model(inputs=input_layer, outputs=output_layer)

		self._compile(model, config.optimizer, config.loss)

		if self.__summarize:
			model.summary()

		return model


class CoreBuilder(ModelBuilder):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, output_activation=sigmoid, **kwargs)

	def _get_input_shape(self, seq_len: int) -> int:
		return seq_len + 1

	def _compile(self, model: Model, optimizer: keras.optimizers.Optimizer, loss: Callable):
		model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


class DeltaBuilder(ModelBuilder):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, un_norm=True, **kwargs)

	def _get_input_shape(self, seq_len: int) -> int:
		return seq_len + 2