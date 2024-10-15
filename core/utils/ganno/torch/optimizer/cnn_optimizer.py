import random
import typing

from core import Config
from .optimizer import Optimizer
from ..choice_utils import ChoiceUtils
from ..nnconfig import CNNConfig, ConvLayer, ModelConfig
from .linear_optimizer import LinearOptimizer


class CNNOptimizer(Optimizer):

	def __init__(self, vocab_size, dataset, test_dataset, *args, input_size=Config.INPUT_SIZE, **kwargs):
		super().__init__(vocab_size, dataset, test_dataset, *args, **kwargs)
		self.__linear_optimizer = LinearOptimizer(
			dataset=dataset,
			test_dataset=test_dataset,
			vocab_size=self._vocab_size,
		)
		self.__input_size = input_size

	def __generate_random_cnn_config(self) -> CNNConfig:
		num_layers = random.randint(3, 8)
		layers = []
		input_size = self.__input_size
		for i in range(num_layers):
			kernel_floor, kernel_ceil = 3, min(7, input_size)
			if kernel_ceil <= kernel_floor:
				break
			kernel_size = random.randint(3, min(7, input_size))
			features = random.randint(16, 256)
			padding = 0
			if input_size - kernel_size > 0:
				padding = random.randint(0, min(3, input_size - kernel_size))
			pooling = 0
			if input_size - kernel_size - 2 * padding > 0:
				pooling = random.randint(0, min(2, input_size - kernel_size - 2 * padding))

			layer = ConvLayer(
				kernel_size,
				features,
				padding,
				pooling,
				norm=ChoiceUtils.choice_discrete(True, False)
			)
			layers.append(layer)
			# Update input size for next layer
			input_size = (input_size - kernel_size + 2 * padding) // (pooling + 1)
		dropout = random.uniform(0, 0.5)
		config = CNNConfig(
			vocab_size=self._vocab_size,
			layers=layers,
			dropout=dropout,
			ff_block=self.__linear_optimizer.generate_random_config()
		)
		return config

	def _generate_initial_generation(self) -> typing.List[ModelConfig]:
		return [
			self.__generate_random_cnn_config()
			for _ in range(int(self._population_size))
		]
