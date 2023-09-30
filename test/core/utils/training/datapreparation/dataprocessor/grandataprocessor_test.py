import unittest
from unittest.mock import MagicMock

import numpy as np
from tensorflow import keras

from core.utils.ganno.builder import GranModelBuilder
from core.utils.ganno.nnconfig import ModelConfig, KalmanFiltersConfig
from core.utils.training.datapreparation.dataprocessor.gran_dataprocessor import GranDataProcessor


class GranDataProcessorTest(unittest.TestCase):

	def __create_model(self):
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

		return builder.build(config)

	def test_functionality(self):
		generator = MagicMock()
		generator.__getitem__ = MagicMock(return_value=np.arange(1100).reshape((10, 110)))
		generator.__len__ = MagicMock(return_value=30)

		grans = sorted(list(np.concatenate([
			1 + bound * np.linspace(-1, 1, size)**pow
			for bound, size, pow in [
				(4e-3, 5, 3)
			]
		])))

		model = self.__create_model()
		processor = GranDataProcessor(
			grans,
			model,
			generator,
			32,
			32
		)
		for i in range(5):
			element = processor.get_data(0, i)
			self.assertTrue(isinstance(element, tuple))
