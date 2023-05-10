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
			ara_tries: int = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._num_actions = num_actions
		self.__model = self._init_ara_model()
		self.__batch = []
		self.__batch_size = batch_size
		if ara_tries is None:
			ara_tries = num_actions*10
		self.__tries = ara_tries

	@abstractmethod
	def _init_ara_model(self) -> Model:
		pass

	@abstractmethod
	def _prepare_input(self, state: object, index: int) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_output(self, state: object, output: typing.List[np.ndarray]) -> object:
		pass

	@abstractmethod
	def _prepare_train_output(self, state: object, action: object) -> typing.List[np.ndarray]:
		pass

	def _generate_action(self, inputs: np.ndarray) -> typing.List[np.ndarray]:
		return self.__model.predict(inputs)

	def _generate_actions(self, state) -> typing.List[object]:

		actions = []
		i = 0
		while len(actions) < self._num_actions and i < self.__tries:
			action = self._prepare_output(
				state,
				self._generate_action(
					self._prepare_input(state, i)
				)
			)
			if self._get_environment().is_action_valid(action, state) and action not in actions:
				actions.append(action)
			i += 1
		return actions

	def _prepare_train_data(
			self,
			batch: typing.List[typing.Tuple[object, object, float]]
	) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
		# TODO: YOU ARE HERE: CONVERT _prepare_* to batch based so to recommend asynchronously.

		states = list(set([instance[0] for instance in batch]))
		X, y = [], None
		for state in states:
			state_instances = sorted(
				[instance for instance in batch if instance[0] == state],
				key=lambda instance: instance[2],
				reverse=True
			)
			for i, instance in enumerate(state_instances):
				X.append(self._prepare_input(instance[0], i))
				outs = self._prepare_train_output(instance[0], instance[1])
				if y is None:
					y = [[] for _ in outs]
				for j, out in enumerate(outs):
					y[j].append(out)

		return np.array(X), [np.array(o) for o in y]

	def _fit_model(self, model: Model, X: np.ndarray, y: typing.List[np.ndarray]):
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


class ActionRecommendationBalancerAgent(ActionRecommendationAgent, ABC):

	def __init__(self, num_actions: float, *args, recommendation_percent: float = 0.5, **kwargs):
		super().__init__(*args, num_actions=int(num_actions*recommendation_percent), **kwargs)

	@abstractmethod
	def _generate_static_actions(self, state: object) -> typing.List[object]:
		pass

	def __select_actions(
			self,
			static: typing.List[object],
			recommended: typing.List[object]
	) -> typing.List[object]:
		return recommended + static[:self._num_actions - len(recommended)]

	def _generate_actions(self, state) -> typing.List[object]:
		return self.__select_actions(
			self._generate_static_actions(state),
			super()._generate_actions(state)
		)
