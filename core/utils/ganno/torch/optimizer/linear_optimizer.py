import typing

from .optimizer import Optimizer
from ..choice_utils import ChoiceUtils
from ..nnconfig import ModelConfig, LinearConfig


class LinearOptimizer(Optimizer):

	def __init__(
			self,
			*args,
			layers_range: typing.Tuple[int, int] = (0, 32),
			layer_size_range: typing.Tuple[int, int] = (32, 2048),
			input_size: typing.Tuple[int, ...] = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__input_size = input_size
		self.__layers_range, self.__layer_size_range = layers_range, layer_size_range

	def generate_random_config(self) -> LinearConfig:
		layers = ChoiceUtils.generate_list(
			*self.__layer_size_range,
			size=self.__layers_range
		)
		return LinearConfig(
			vocab_size=self._vocab_size,
			layers=layers,
			dropout=ChoiceUtils.choice_continuous(0, 1,),
			norm=ChoiceUtils.generate_list(True, False, size=len(layers)),
			block_size=self.__input_size
		)

	def _generate_initial_generation(self) -> typing.List[ModelConfig]:
		return [
			self.generate_random_config()
			for _ in range(int(self._population_size))
		]
