import tensorflow as tf
from tensorflow.keras.layers import Layer


class FloatEmbedding(Layer):

	def __init__(self, units, **kwargs):
		self.units = units
		super(FloatEmbedding, self).__init__(**kwargs)

	def get_config(self):
		return {
			"units": self.units
		}

	def build(self, input_shape):
		if len(input_shape) != 3:
			raise Exception(f"Expected rank 3 but got input_shape={input_shape}")
		self.w = self.add_weight(
			shape=(input_shape[2], self.units),
			initializer="random_normal",
			trainable=True,
			name="embedding_weights"
		)
		self.b = self.add_weight(
			shape=(self.units,),
			initializer="random_normal",
			trainable=True,
			name="embedding_bias"
		)

	def call(self, inputs, **kwargs):
		return tf.matmul(inputs, self.w) + self.b


class PositionalEncoding(Layer):

	def __init__(self, **kwargs):
		super(PositionalEncoding, self).__init__(**kwargs)

	def call(self, inputs, **kwargs):
		# inputs.shape[2] must be an even number
		r = tf.cast(tf.reshape(tf.range(inputs.shape[1]), (-1, 1)), tf.float64) / (
				10000 ** (2 * tf.reshape(tf.range(inputs.shape[2]), (1, -1)) / inputs.shape[2]))
		return tf.cast(inputs, tf.float64) + tf.reshape(tf.concat([tf.reshape(tf.sin(r[:,::2]), (r.shape[0], -1, 1)), tf.reshape(tf.sin(r[:,1::2]), (r.shape[0], -1, 1))], axis=2), r.shape)


class BiasLayer(Layer):

	def __init__(self, **kwargs):
		super(BiasLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.b = self.add_weight(name="bias",
								shape=(1,),
								trainable=True,
								initializer="zeros"
								)

	def call(self, inputs, **kwargs):
		return inputs + self.b
