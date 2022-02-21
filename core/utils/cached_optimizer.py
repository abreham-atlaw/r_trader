from typing import *
from abc import ABC

from lib.dnn.utils import Optimizer, SampledOptimizer

from .optimization_cacher.OptimizationCacher import OptimizationCacher


class CachedOptimizer(SampledOptimizer, ABC):

	def __init__(self, user: str, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__user = user
		self.__cacher = OptimizationCacher()
		self.__restart_required: bool = True

	def _get_value_loss(self, values: Dict):
		if self.__cacher.is_locked(values):
			print(f"[+]Params is locked. Skipping...")
			self.__restart_required = True
			return None
		cached = self.__cacher.get_value(values)
		if cached is not None:
			print(f"[+]Returning Cached Value for {values}: {cached}")
			return cached
		self.__cacher.lock(values, self.__user)
		value = super()._get_value_loss(values)
		self.__cacher.cache(values, value)
		return value

	def optimize(self):
		return_value = None
		while self.__restart_required:
			self.__restart_required = False
			return_value = super().optimize()
		return return_value
