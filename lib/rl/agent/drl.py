import typing
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras.models import Model

from lib.rl.agent.utils.agent_data_generator import AgentDataGenerator
from lib.rl.agent import ActionChoiceAgent


class DeepReinforcementAgent(ActionChoiceAgent, ABC):

	def __init__(self, *args, batch_size=1, **kwargs):
		super().__init__(*args, **kwargs)
		self.__model = self._init_model()
		self.__generator = AgentDataGenerator(batch_size)

	@abstractmethod
	def _init_model(self) -> Model:
		pass

	@abstractmethod
	def _prepare_dra_input(self, state: typing.Any, action: typing.Any) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_dra_output(self, state: typing.Any, action: typing.Any, output: np.ndarray) -> float:
		pass

	@abstractmethod
	def _prepare_train_value(self, state: typing.Any, action: typing.Any, value: float) -> np.ndarray:
		pass

	def _fit_model(self, model: Model, X: np.ndarray, y: np.ndarray):
		self.__model.fit(X, y)

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		return self._prepare_dra_output(
			state,
			action,
			np.squeeze(
				self.__model.predict(
					np.expand_dims(
						self._prepare_dra_input(
							state,
							action
						),
						0
					)
				)
			)
		)

	def _update_state_action_value(self, initial_state, action, final_state, value):
		self.__generator.append(
			self._prepare_dra_input(initial_state, action),
			self._prepare_train_value(initial_state, action, value)
		)
		if len(self.__generator) == 2:
			X, y = self.__generator[0]
			self._fit_model(self.__model, X, y)
			self.__generator.remove(0)
