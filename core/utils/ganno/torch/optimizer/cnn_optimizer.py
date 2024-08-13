import random
import typing

from .optimizer import Optimizer
from ..nnconfig import CNNConfig, ConvLayer, ModelConfig


class CNNOptimizer(Optimizer):

	@staticmethod
	def __generate_random_cnn_config(vocab_size: int) -> CNNConfig:
		num_layers = random.randint(3, 8)
		layers = []
		input_size = vocab_size
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
			layer = ConvLayer(kernel_size, features, padding, pooling)
			layers.append(layer)
			# Update input size for next layer
			input_size = (input_size - kernel_size + 2 * padding) // (pooling + 1)
		dropout = random.uniform(0, 0.5)
		config = CNNConfig(vocab_size, layers, dropout)
		return config

	def _generate_initial_generation(self) -> typing.List[ModelConfig]:
		return [
			self.__generate_random_cnn_config(self._vocab_size)
			for _ in range(int(self._population_size))
		]
