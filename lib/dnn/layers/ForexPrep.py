import tensorflow as tf

from tensorflow.keras.layers import Layer, Concatenate, Reshape, Activation


class Delta(Layer):

	def __init__(self, name="delta", **kwargs):
		super(Delta, self).__init__(name=name, **kwargs)

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
		return (inputs - tf.reshape(min_, (-1, 1))) / tf.reshape(tf.reduce_max(inputs, axis=1) - min_, (-1, 1))
		# return (inputs - tf.reduce_min(inputs, axis=1))/tf.reshape(tf.reduce_max([tf.reduce_max(inputs, axis=1), tf.abs(tf.reduce_min(inputs, axis=1))], axis=0), (-1, 1, 1))


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


class MovingAverage(Layer):

	def __init__(self, average_gap, name="moving_average", **kwargs):
		self.average_gap = average_gap
		super(MovingAverage, self).__init__(name=name, **kwargs)

	@tf.function
	def calc_moving_average(self, inputs):
		output = []
		for i in range(inputs.shape[1]-self.average_gap+1):
			output.append(
				tf.reduce_mean(inputs[:, i: self.average_gap+i], axis=1)
			)
		return tf.stack(output, axis=1)

	def call(self, inputs, **kwargs):
		return self.calc_moving_average(inputs)

	def get_config(self):
		return {
			"average_gap": self.average_gap
		}


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
