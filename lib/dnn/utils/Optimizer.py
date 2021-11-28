from typing import *
from abc import ABC, abstractmethod

import numpy as np

import gc

from lib.dnn.utils import KerasTrainer


class Optimizer(ABC):

	def __init__(self):
		self.__param_values: Dict = self._generate_param_values()

	@abstractmethod
	def _generate_param_values(self) -> Dict:
		pass

	@abstractmethod
	def _create_trainer(self, params) -> KerasTrainer:
		pass

	def __get_param_values(self, param) -> List:
		return self.__param_values[param]

	def __get_params(self) -> List[str]:
		return list(self.__param_values.keys())

	def _get_value_loss(self, values: Dict) -> float:
		print(f"[+]Getting Loss for :{values}")
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

	def _optimize_params(self, params: List[str], default_values: Dict) -> Tuple[Dict, float]:
		print(f"[+]Optimizing Params: {params} with Default Values: {default_values}")

		min_loss = None
		optimal_params = None

		is_last_param = len(params) == 1

		for value in self.__get_param_values(params[0]):
			new_values = default_values.copy()
			new_values[params[0]] = value
			if is_last_param:
				loss = self._get_value_loss(new_values)
				if loss is None:
					continue
			else:
				candidate_values, loss = self._optimize_params(params[1:], new_values)
			print(min_loss, loss)
			if min_loss is None or loss < min_loss:
				if is_last_param:
					optimal_params = new_values
				else:
					optimal_params = candidate_values
				min_loss = loss

		return optimal_params, min_loss

	def optimize(self):
		print("[+]Starting Optimization...")
		return self._optimize_params(self.__get_params(), {})


