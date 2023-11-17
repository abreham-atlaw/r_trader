import unittest

from tensorflow import keras

from core.utils.ganno.builder import GranModelBuilder
from core.utils.ganno.nnconfig import ModelConfig, KalmanFiltersConfig


class BuilderTest(unittest.TestCase):

	def test_gran_model_builder(self):
		builder = GranModelBuilder(
			5,
		)
		config = ModelConfig(
			seq_len=64,
			ff_dense_layers=[
				(64, 0.2),
				(32, 0)
			],
			ff_conv_pool_layers=[

			],
			delta=True,
			norm=True,
			rsi=[],
			wpr=[],
			kalman_filters=KalmanFiltersConfig(
				16,
				[]
			),
			kalman_static_filters=[],
			mas_windows=[],
			msd_windows=[],
			trend_lines=[],
			include_prep=True,
			stochastic_oscillators=[14],
			conv_activation=keras.activations.relu,
			dense_activation=keras.activations.relu,
			loss=keras.losses.binary_crossentropy,
			optimizer=keras.optimizers.Adam(),
		)

		model: keras.models.Model = builder.build(config)
		self.assertEqual(model.output_shape, (None, 6))
