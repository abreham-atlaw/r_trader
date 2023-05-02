import typing
from abc import ABC, abstractmethod

import numpy as np
from tensorflow.keras.models import Model

from .action_choice_agent import ActionChoiceAgent


class ActionRecommendationAgent(ActionChoiceAgent, ABC):

	def __init__(
			self,
			num_actions: int,
			*args,
			batch_size: int = 32,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._num_actions = num_actions
		self.__model = self._init_ara_model()
		self.__batch = []
		self.__batch_size = batch_size

	@abstractmethod
	def _init_ara_model(self) -> Model:
		pass

	@abstractmethod
	def _prepare_input(self, state: object, index: int) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_output(self, state: object, output: np.ndarray) -> object:
		pass

	@abstractmethod
	def _prepare_train_output(self, state: object, action: object) -> np.ndarray:
		pass

	def _generate_action(self, inputs: np.ndarray) -> np.ndarray:
		return self.__model.predict(inputs)

	def _generate_actions(self, state) -> typing.List[object]:
		return [
			self._prepare_output(
				state,
				self._generate_action(
					self._prepare_input(state, i)
				)
			)
			for i in range(self._num_actions)
		]

	def _prepare_train_data(
			self,
			batch: typing.List[typing.Tuple[object, object, float]]
	) -> typing.Tuple[np.ndarray, np.ndarray]:
		states = list(set([instance[0] for instance in batch]))
		X, y = [], []
		for state in states:
			state_instances = sorted(
				[instance for instance in batch if instance[0] == state],
				key=lambda instance: instance[2],
				reverse=True
			)
			for i, instance in range(len(state_instances)):
				X.append(self._prepare_input(instance[0], i))
				y.append(self._prepare_train_output(instance[0], instance[1]))

		return np.array(X), np.array(y)

	def _fit_model(self, model: Model, X: np.ndarray, y: np.ndarray):
		model.fit(X, y)

	def _train_batch(self, batch: typing.List[typing.Tuple[object, object, float]], model: Model):
		X, y = self._prepare_train_data(batch)
		self._fit_model(model, X, y)

	def __update_instance(self, state, action, value):
		self.__batch.append((state, action, value))
		if len(self.__batch) >= self.__batch_size:
			self._train_batch(self.__batch, self.__model)
			self.__batch = []

	def _update_state_action_value(self, initial_state, action, final_state, value):
		self.__update_instance(initial_state, action, value)
		super()._update_state_action_value(initial_state, action, final_state, value)
