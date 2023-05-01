import typing
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras.models import Model

from .action_choice_agent import ActionChoiceAgent


class ActionRecommendationAgent(ActionChoiceAgent, ABC):

	def __init__(self, ara_model: Model, generation_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__ara_model = ara_model
		self.__generation_size = generation_size

	@abstractmethod
	def _prepare_input(self, state, index: int) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_output(self, output: np.ndarray) -> object:
		pass

	def __generate_action(self, state: np.ndarray, index: int) -> object:
		inputs = self._prepare_input(state, index)
		output = self.__ara_model.predict(inputs)
		action = self._prepare_output(output)
		return action

	def _generate_actions(self, state) -> typing.List[object]:
		actions = [
			self.__generate_action(
				state,
				i
			)
			for i in range(self.__generation_size)
		]
		return list(filter(
			lambda action: self._get_environment().is_action_valid(action, state),
			actions
		))
