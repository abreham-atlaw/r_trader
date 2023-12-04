import random
import typing

from .optimizer import Optimizer
from ..nnconfig import CNNConfig, ConvLayer, ModelConfig, LinearConfig


class LinearOptimizer(Optimizer):

	def __init__(
			self,
			*args,
			layers_range: typing.Tuple[int, int] = (32, 2048),
			layer_size_range: typing.Tuple[int, int] = (3, 24),
			dropout_range: typing.Tuple[int, int] = (0, 0.5),
			block_size: int = 1024,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__layers_range, self.__layer_size_range, self.__dropout_range = layers_range, layer_size_range, dropout_range
		self.__block_size = block_size

	def __generate_random_config(self) -> LinearConfig:
		num_layers = random.randint(*self.__layers_range)
		layers = [random.randint(*self.__layer_size_range) for _ in range(num_layers)]
		dropout = random.uniform(*self.__dropout_range)
		return LinearConfig(
			vocab_size=self._vocab_size,
			layers=layers,
			dropout=dropout,
			block_size=self.__block_size
		)

	def _generate_initial_generation(self) -> typing.List[ModelConfig]:
		return [
			self.__generate_random_config()
			for _ in range(int(self._population_size))
		]
