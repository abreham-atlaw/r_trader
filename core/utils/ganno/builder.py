import math
from typing import *
from abc import ABC, abstractmethod

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, MaxPooling1D, Input, Reshape, Concatenate, Flatten, Dropout, \
	Add, Subtract, AveragePooling1D, UpSampling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.activations import sigmoid
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from lib.utils.logger import Logger
from lib.dnn.layers import Delta, Norm, UnNorm, StochasticOscillator, TrendLine, OverlayIndicator,\
	WilliamsPercentageRange, RelativeStrengthIndex, MovingAverage, MovingStandardDeviation, OverlaysCombiner, KalmanFilter,\
	KalmanStaticFilter, FloatEmbedding, PositionalEncoding
from .nnconfig import ModelConfig, ConvPoolLayer, KalmanFiltersConfig, TransformerConfig


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
	def _add_ff_dense_layers(layer: KerasTensor, layers: List[Tuple[int, float]], activation: Callable) -> KerasTensor:
		for layer_size, dropout in layers:
			if layer_size != 0:
				layer = Dense(layer_size, activation=activation)(layer)
			if dropout != 0:
				layer = Dropout(dropout)(layer)
		return layer

	@staticmethod
	def _add_ff_conv_layers(layer: KerasTensor, layers: List[ConvPoolLayer], activation: Callable) -> KerasTensor:
		for config in layers:
			if config.size != 0:
				layer = Conv1D(
					kernel_size=config.size,
					filters=config.features,
					activation=activation,
					padding=config.padding
				)(layer)
			if config.pool != 0:
				layer = MaxPooling1D(pool_size=config.pool)(layer)
			if config.dropout != 0:
				layer = Dropout(config.dropout)(layer)

		return layer

	@staticmethod
	def _add_transformer_block(layer: KerasTensor, config: TransformerConfig) -> KerasTensor:
		attention = MultiHeadAttention(config.heads, key_dim=layer.shape[-1])(layer, layer)
		attention = Dropout(config.attention_dropout)(attention)
		norm = LayerNormalization()(Add()([attention, layer]))
		ff_out = ModelBuilder._add_ff_dense_layers(norm, config.ff_dense + [(norm.shape[2], config.dense_dropout)], config.dense_activation)
		return LayerNormalization()(Add()([ff_out, norm]))

	@staticmethod
	def __create_kalman_filter(input_layer: KerasTensor, compute_size: int, percentage: float, initial_size: int) -> KerasTensor:
		input_layer = input_layer[:, -math.floor(percentage * initial_size):]
		pool_size = math.ceil(input_layer.shape[1]/compute_size)
		if pool_size > 1:
			input_layer = Flatten()(
				AveragePooling1D(pool_size)(
					Reshape((-1, 1))(input_layer)
				)
			)
		out = KalmanFilter()(input_layer)
		if pool_size > 1:
			out = Flatten()(
				UpSampling1D(math.ceil(pool_size))(
					Reshape((-1, 1))(out)
				)
			)
		return out

	@staticmethod
	def _add_kalman_filters(layer: KerasTensor, config: KalmanFiltersConfig) -> List[KerasTensor]:

		filters = [
			ModelBuilder.__create_kalman_filter(
				layer,
				config.compute_size,
				config.percentages[0],
				layer.shape[1]
			)
		]
		filters_sum = filters[0]
		filters_size = filters_sum.shape[1]
		for p in config.percentages[1:]:
			filters.append(
				ModelBuilder.__create_kalman_filter(
					Subtract()((layer[:, -filters_size:], filters_sum)),
					config.compute_size,
					p,
					layer.shape[1]
				)
			)
			filters_size = min(filters_sum.shape[1], filters[-1].shape[1])
			filters_sum = Add()((filters_sum[:, -filters_size:], filters[-1][:, -filters_size]))

		return filters

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

		overlays = []
		if config.include_prep:
			overlays.append(prep_layer)
		for cls, args in [
			(StochasticOscillator, config.stochastic_oscillators),
			(RelativeStrengthIndex, config.rsi),
			(WilliamsPercentageRange, config.wpr),
			(MovingAverage, config.mas_windows),
			(MovingStandardDeviation, config.msd_windows),
			(KalmanStaticFilter, config.kalman_static_filters),
			(Delta, [()][:int(config.delta)]),
		]:
			overlays.extend(self.__create_overlays(cls, args, prep_layer))

		if len(config.kalman_filters.percentages) > 0:
			overlays.extend(self._add_kalman_filters(prep_layer, config.kalman_filters))

		combined = OverlaysCombiner()(overlays)

		ff_conv = self._add_ff_conv_layers(combined, config.ff_conv_pool_layers, config.conv_activation)

		embedding = ff_conv
		if config.float_embedding is not None:
			embedding = FloatEmbedding(config.float_embedding)(embedding)

		encoding = embedding
		if config.positional_encoding:
			encoding = PositionalEncoding()(encoding)

		transformer_block = encoding
		if config.transformer_config is not None:
			transformer_block = self._add_transformer_block(encoding, config.transformer_config)

		trend_lines = self.__concat_layers(
			input_sequence,
			TrendLine,
			[(arg,) for arg in config.trend_lines]
		)

		flatten_inputs = [Flatten()(transformer_block), trend_lines, extra_input]
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
