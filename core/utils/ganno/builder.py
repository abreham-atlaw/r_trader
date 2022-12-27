from typing import *
from abc import ABC, abstractmethod

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, MaxPooling1D, Input, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.activations import sigmoid
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from lib.utils.logger import Logger
from lib.dnn.layers import Delta, Norm, UnNorm, StochasticOscillator, TrendLine, OverlayIndicator,\
	WilliamsPercentageRange, RelativeStrengthIndex, MovingAverage, MovingStandardDeviation, OverlaysCombiner, KelmanFilter,\
	KelmanStaticFilter
from .nnconfig import ModelConfig, ConvPoolLayer


class ModelBuilder(ABC):

	def __init__(self, un_norm=False, output_activation=None, summarize=True):
		self.__un_norm = un_norm
		self.__output_activation = output_activation
		self.__summarize = summarize

	@staticmethod
	def __concat_layers(inputs: Layer, layer_class: Type, args: List[Any], axis=1) -> Optional[Layer]:
		if len(args) == 0:
			return None
		layers = [layer_class(*arg)(inputs) for arg in args]
		return Concatenate(
			axis=axis
		)(layers)

	@staticmethod
	def _add_ff_dense_layers(layer: KerasTensor, layers: List[Tuple[int, int]], activation: Callable) -> Layer:
		for layer_size, dropout in layers:
			if layer_size != 0:
				layer = Dense(layer_size, activation=activation)(layer)
			if dropout != 0:
				layer = Dropout(dropout)(layer)
		return layer

	@staticmethod
	def _add_ff_conv_layers(layer: KerasTensor, layers: List[ConvPoolLayer], activation: Callable) -> Layer:
		for config in layers:
			if config.size != 0:
				layer = Conv1D(kernel_size=config.size, filters=config.features, activation=activation)(layer)
			if config.pool != 0:
				layer = MaxPooling1D(pool_size=config.pool)(layer)
			if config.dropout != 0:
				layer = Dropout(config.dropout)(layer)

		return layer

	@staticmethod
	def __create_overlays(cls: Type, args: List[Tuple], inputs: Layer) -> List[OverlayIndicator]:
		if len(args) == 0:
			return []
		if not isinstance(args[0], tuple):
			args = [(arg,) for arg in args]

		return [
			cls(*arg)(inputs)
			for arg in args
		]

	@abstractmethod
	def _get_input_shape(self, config: ModelConfig) -> int:
		pass

	def _get_output_shape(self, config: ModelConfig) -> int:
		return 1

	def _finalize_output_layer(self, output_layer: KerasTensor, config: ModelConfig):
		return output_layer

	def _compile(self, model: Model, optimizer: keras.optimizers.Optimizer, loss: Callable):
		model.compile(optimizer=optimizer, loss=loss)

	def _build(self, config: ModelConfig) -> Model:
		Logger.info("[+]Building", config)
		input_layer = Input(shape=self._get_input_shape(config))

		input_sequence = input_layer[:, :config.seq_len]
		extra_input = input_layer[:, config.seq_len:]

		prep_layer = input_sequence
		if config.norm:
			prep_layer = Norm()(prep_layer)

		overlays = [prep_layer]
		for cls, args in [
			(StochasticOscillator, config.stochastic_oscillators),
			(RelativeStrengthIndex, config.rsi),
			(WilliamsPercentageRange, config.wpr),
			(MovingAverage, config.mas_windows),
			(MovingStandardDeviation, config.msd_windows),
			(KelmanStaticFilter, config.kelman_static_filters),
			(KelmanFilter, [() for _ in range(config.kelman_filters)]),
			(Delta, [()][:int(config.delta)]),
		]:
			overlays.extend(self.__create_overlays(cls, args, prep_layer))

		combined = OverlaysCombiner()(overlays)

		ff_conv = self._add_ff_conv_layers(combined, config.ff_conv_pool_layers, config.conv_activation)

		trend_lines = self.__concat_layers(
			input_sequence,
			TrendLine,
			[(arg,) for arg in config.trend_lines]
		)

		flatten_inputs = [Flatten()(ff_conv), trend_lines, extra_input]
		if trend_lines is None:
			flatten_inputs.pop(1)

		flatten = Concatenate(axis=1)(flatten_inputs)

		ff_dense = self._add_ff_dense_layers(flatten, config.ff_dense_layers, config.dense_activation)

		output_layer = Dense(
			self._get_output_shape(config),
			activation=self.__output_activation
		)(ff_dense)

		if self.__un_norm:
			output_layer = UnNorm(min_increment=False)(
				Concatenate(axis=1)((
					input_sequence,
					output_layer
				))
			)
			output_layer = Reshape((1,))(output_layer)

		model = Model(inputs=input_layer, outputs=self._finalize_output_layer(output_layer, config))

		self._compile(model, config.optimizer, config.loss)

		if self.__summarize:
			model.summary()

		return model

	def build(self, config: ModelConfig) -> Model:
		try:
			return self._build(config)
		except Exception as ex:
			raise BuildException(str(ex))


class CoreBuilder(ModelBuilder):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, output_activation=sigmoid, **kwargs)

	def _get_input_shape(self, config: ModelConfig) -> int:
		return config.seq_len + 1

	def _compile(self, model: Model, optimizer: keras.optimizers.Optimizer, loss: Callable):
		model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


class DeltaBuilder(ModelBuilder):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, un_norm=True, **kwargs)

	def _get_input_shape(self, config: ModelConfig) -> int:
		return config.seq_len + 2


class BuildException(Exception):
	pass
