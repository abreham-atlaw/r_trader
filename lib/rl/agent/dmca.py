import typing
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras import models

from .mca import MonteCarloAgent


class DeepMonteCarloAgent(MonteCarloAgent, ABC):

	def __init__(self, *args, dmca_exploit_rate=0.7, **kwargs):
		super().__init__(*args, **kwargs)
		self.__dmca_exploit_rate = dmca_exploit_rate

	@abstractmethod
	def _init_mc_dnn(self) -> models.Model:
		pass

	@abstractmethod
	def _prepare_dmca_input(self, state: typing.Any) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_dmca_output(self, state: typing.Any, output: np.ndarray) -> float:
		pass

	@abstractmethod
	def _prepare_train_output(self, state: typing.Any, action: typing.Any, value: float) -> np.ndarray:
		pass

	def _explore(self, state_node: MonteCarloAgent.Node):
		pass

	def _simulate(self, state_node: MonteCarloAgent.Node):




