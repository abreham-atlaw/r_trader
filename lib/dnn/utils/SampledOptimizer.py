from typing import *
from abc import ABC, abstractmethod

import numpy as np

import gc

from lib.utils.logger import Logger
from lib.dnn.utils import KerasTrainer


class SampledOptimizer(ABC):

	def __init__(self):
		self.__samples = None

	@abstractmethod
	def _get_samples(self) -> List[Dict]:
		pass

	@abstractmethod
	def _create_trainer(self, params) -> KerasTrainer:
		pass

	@Logger.logged_method
	def _get_value_loss(self, values: Dict) -> float:
		trainer = self._create_trainer(values)
		train_history, test_loss = trainer.start()
		if len(test_loss) > 1:
			test_loss = test_loss[0]
		del trainer

		gc.collect()

		return np.average([
			np.min(train_history.history["loss"]),
			test_loss
		])

	@Logger.logged_method
	def _optimize_params(self, param_values: List[Dict]) -> Tuple[Dict, float]:

		min_loss = None
		optimal_params = None
		for values in param_values:
			loss = self._get_value_loss(values)
			if min_loss is None or loss < min_loss:
				min_loss = loss
				optimal_params = values

		return optimal_params, min_loss

	def optimize(self) -> Tuple[Dict, float]:
		return self._optimize_params(self._get_samples())
