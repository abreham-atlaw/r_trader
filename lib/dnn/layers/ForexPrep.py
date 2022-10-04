from typing import *
from abc import abstractmethod, ABC

import tensorflow as tf

from tensorflow.keras.layers import Layer, Concatenate, Reshape, Activation


class Sign(Layer):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def call(self, inputs, *args, **kwargs):
		return tf.math.divide_no_nan(inputs, tf.abs(inputs))


class SignFilter(Layer):

	def __init__(self, sign, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__sign = sign
		self.__sign_layer = Sign()

	def call(self, inputs, *args, **kwargs):
		return tf.abs(
			tf.round(
				(self.__sign_layer(inputs) + self.__sign)/2
			)
		)*inputs


class Delta(Layer):

	def __init__(self, **kwargs):
		super(Delta, self).__init__( **kwargs)

	def call(self, inputs: tf.Tensor, **kwargs):
		return inputs[:, 1:] - inputs[:, :-1]


class Percentage(Layer):

	def __init__(self, **kwargs):
		super(Percentage, self).__init__(**kwargs)

	def call(self, inputs, **kwargs):
		if inputs.shape[1] != 2:
			raise Exception(f"Incompatible Shape {inputs.shape}")
		return inputs[:, 0]/inputs[:, 1]


class Norm(Layer):

	def __init__(self,  **kwargs):
		super(Norm, self).__init__(**kwargs)

	def call(self, inputs, **kwargs):
		min_ = tf.reduce_min(inputs, axis=1)
		return tf.math.divide_no_nan(
			(inputs - tf.reshape(min_, (-1, 1))),
			tf.reshape(tf.reduce_max(inputs, axis=1) - min_, (-1, 1))
		)


class UnNorm(Layer):

	def __init__(self, min_increment=True,  **kwargs):
		super(UnNorm, self).__init__(**kwargs)
		if min_increment:
			self.min_increment = 1
		else:
			self.min_increment = 0

	def get_config(self):
		config = super(UnNorm, self).get_config()
		config["min_increment"] = (self.min_increment == 1)
		return config

	def call(self, inputs, *args, **kwargs):
		raw = inputs[:, :-1]
		norm = inputs[:, -1]
		min_ = tf.reduce_min(raw, axis=1)
		max_ = tf.reduce_max(raw, axis=1)

		return (norm * (max_ - min_)) + (min_*self.min_increment)


class OverlayIndicator(Layer, ABC):

	def __init__(self, window_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__window_size = window_size

	@abstractmethod
	def _on_time_point(self, inputs: tf.Tensor) -> tf.Tensor:
		pass

	@tf.function
	def call(self, inputs, *args, **kwargs):
		output = []
		for i in range(inputs.shape[1] - self.__window_size+1):
			output.append(
				self._on_time_point(inputs[:, i:self.__window_size+i])
			)
		return tf.stack(output, axis=1)

	def get_config(self):
		return {
			"window_size": self.__window_size
		}


class MultipleMovingAverages(Layer):

	def __init__(self, sizes: List[int], *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__out_size_decrement = max(sizes) - 1
		self.__mas = [MovingAverage(size) for size in sizes]

	def call(self, inputs, *args, **kwargs):
		out_size = inputs.shape[1] - self.__out_size_decrement
		return tf.stack([
			ma(inputs)[:, -out_size:]
			for ma in self.__mas
		], axis=2)


class OverlaysCombiner(Layer):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def call(self, inputs, *args, **kwargs):
		out_size = min([overlay.shape[1] for overlay in inputs])
		return tf.stack([
			overlay[:, -out_size:]
			for overlay in inputs
		], axis=2)


class MovingAverage(OverlayIndicator):

	def __init__(self, *args, **kwargs):
		super(MovingAverage, self).__init__(*args, **kwargs)

	def _on_time_point(self, inputs: tf.Tensor) -> tf.Tensor:
		return tf.reduce_mean(inputs, axis=1)


class ExponentialMovingAverage(Layer):

	def __init__(self, smoothing_factor):
		super(ExponentialMovingAverage, self).__init__()
		self.__smoothing_factor = smoothing_factor

	@tf.function
	def __calc_ema(self, inputs):
		output = []
		output.append(inputs[:, 0])

		for i in range(1, inputs.shape[1]):
			output.append(self.__smoothing_factor * (inputs[:, i] - output[-1]) + output[-1])

		return tf.stack(output, axis=1)

	def call(self, inputs, *args, **kwargs):
		return self.__calc_ema(inputs)

	def get_config(self):
		return {
			"smoothing_factor": self.__smoothing_factor
		}


class MovingStandardDeviation(OverlayIndicator):

	def __init__(self, *args, **kwargs):
		super(MovingStandardDeviation, self).__init__(*args, **kwargs)

	def _on_time_point(self, inputs: tf.Tensor) -> tf.Tensor:
		return tf.sqrt(
			tf.reduce_sum(
				tf.pow(
					inputs - tf.reshape(tf.reduce_mean(inputs, axis=1), (-1, 1)),
					2
				)/inputs.shape[1],
				axis=1
			)
		)


class WilliamsPercentageRange(OverlayIndicator):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _on_time_point(self, inputs: tf.Tensor) -> tf.Tensor:
		highest = tf.reduce_max(inputs, axis=1)
		lowest = tf.reduce_min(inputs, axis=1)
		return tf.math.divide_no_nan((inputs[:, 0] - highest), (highest - lowest))


class StochasticOscillator(OverlayIndicator):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _on_time_point(self, inputs: tf.Tensor) -> tf.Tensor:
		highest = tf.reduce_max(inputs, axis=1)
		lowest = tf.reduce_min(inputs, axis=1)
		close = inputs[:, 0]
		return tf.math.divide_no_nan((close - lowest), (highest - lowest))


class RelativeStrengthIndex(OverlayIndicator):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__delta = Delta()
		self.__gain_filter = SignFilter(1)
		self.__loss_filter = SignFilter(-1)

	def _on_time_point(self, inputs: tf.Tensor) -> tf.Tensor:
		percentage = tf.math.divide_no_nan(self.__delta(inputs), inputs[:, :-1])
		average_gain = tf.reduce_mean(
			self.__gain_filter(percentage),
			axis=1
		)
		average_loss = tf.reduce_mean(
			-1 * self.__loss_filter(percentage),
			axis=1
		)
		return 1 - (1 / (1 + (average_gain / average_loss)))


class TrendLine(Layer):

	def __init__(self, size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__size = size

	def call(self, inputs, *args, **kwargs):
		return tf.expand_dims(
			(inputs[:, 0] - inputs[:, -self.__size])/self.__size,
			axis=1
		)

	def get_config(self):
		config = super().get_config()
		config["size"] = self.__size
		return config


class ForexPrep(Layer):

	def __init__(self, average_gap=7, **kwargs):
		self.delta = Delta()
		self.concat_reshape = None
		self.concat = Concatenate(axis=1)
		self.percentage = Percentage()
		self.norm = Norm()
		self.tanh = Activation("tanh")
		self.moving_average = MovingAverage(average_gap)
		self.final_concat = Concatenate(axis=2)
		super(ForexPrep, self).__init__(**kwargs)

	def get_config(self):
		return {
			"average_gap": self.moving_average.average_gap
		}

	def build(self, input_shape):
		self.concat_reshape = Reshape(target_shape=(1, input_shape[1]-1, 1))

	def call(self, inputs, **kwargs):

		return self.final_concat([
			self.moving_average(
				self.norm(
					self.percentage(
						self.concat([
							self.concat_reshape(
								self.delta(inputs)
							),
							self.concat_reshape(
								inputs[:, :-1]
							)
						])
					)
				)
			), inputs[:, :-self.moving_average.average_gap]]
		)
