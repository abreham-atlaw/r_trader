import typing

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator


class SampleWeightGeneratorPipeline(AbstractSampleWeightGenerator):

	def __init__(
			self,
			*args,
			generators: typing.List[AbstractSampleWeightGenerator],
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__generators = generators

	def _generate(self, *args, **kwargs) -> np.ndarray:
		output = self.__generators[0](*args, **kwargs)
		for generator in self.__generators[1:]:
			output = generator(output)
		return output
