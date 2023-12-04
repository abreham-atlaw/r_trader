
import random
import typing

from .optimizer import Optimizer
from ..nnconfig import TransformerConfig


class TransformerOptimizer(Optimizer):

	def __init__(self, *args, block_size=1024, **kwargs):
		super().__init__(*args, **kwargs)
		self.__block_size = block_size

	def __generate_random_transformer_config(self, vocab_size: int) -> TransformerConfig:
		kernel_size = random.randint(1, 7)
		num_heads = random.randint(1, 16)
		emb_size = num_heads * random.randint(2, 32)
		block_size = self.__block_size
		ff_size = random.randint(64, 1024)
		config = TransformerConfig(vocab_size, kernel_size, emb_size, block_size, num_heads, ff_size)
		return config

	def _generate_initial_generation(self) -> typing.List[TransformerConfig]:
		return [
			self.__generate_random_transformer_config(self._vocab_size)
			for _ in range(int(self._population_size))
		]
