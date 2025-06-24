import typing

import numpy as np

from .basic_swg import BasicSampleWeightGenerator
from .training.datapreparer import SampleWeightGeneratorDataPreparer
from .training.models import SampleWeightGenerationModel


class GeneratorModelSampleWeightGenerator(BasicSampleWeightGenerator):

	def __init__(
			self,
			*args,
			model: SampleWeightGenerationModel,
			bounds: typing.List[float],
			X_extra_len: int = 124,
			y_extra_len: int = 1,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__model = model
		self.__preparer = SampleWeightGeneratorDataPreparer(
			bounds=bounds,
			X_extra_len=X_extra_len,
			y_extra_len=y_extra_len
		)

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		return self.__model(self.__preparer.prepare(X, y))

