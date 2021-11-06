from typing import *

from lib.dnn.utils import Optimizer

from .optimization_cacher.OptimizationCacher import OptimizationCacher


class CachedOptimizer(Optimizer):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__cacher = OptimizationCacher()

	def _get_value_loss(self, values: Dict):
		cached = self.__cacher.get_value(values)
		if cached is not None:
			return cached
		return super()._get_value_loss(values)
