from typing import *
from abc import ABC, abstractmethod

import numpy as np
from tensorflow import keras

import os

from lib.rl.environment import ModelBasedState
from . import Model
from .. import ModelBasedAgent

TRANSITION_MODEL_FILE_NAME = "transition_model.h5"


class DNNTransitionAgent(ModelBasedAgent, ABC):

	def __init__(self, *args, batch_update=True, update_batch_size=100, **kwargs):
		super().__init__(*args, **kwargs)
		self._enable_batch_update = batch_update
		self._update_batch_size = update_batch_size
		if self._enable_batch_update:
			self._update_batch = ([], [])
			
		self.__transition_model: Union[Model or None] = None
		self.__cache = {}

	@abstractmethod
	def _state_action_to_model_input(self, state: ModelBasedState, action, final_state: ModelBasedState) -> np.ndarray:
		pass

	@abstractmethod
	def _prediction_to_transition_probability(self, initial_state: ModelBasedState, output: np.ndarray, final_state: ModelBasedState) -> float:
		pass

	@abstractmethod
	def _get_train_output(self, initial_state: ModelBasedState, action, final_state: ModelBasedState) -> np.ndarray:
		pass

	@property
	def _transition_model(self) -> Model:
		if self.__transition_model is None:
			raise Exception("Transition Model not Set")
		return self.__transition_model

	def set_transition_model(self, model: Model):
		self.__transition_model = model

	def _get_expected_transition_probability(self, initial_state: ModelBasedState, action, final_state: ModelBasedState) -> float:
		prediction_input = self._state_action_to_model_input(initial_state, action, final_state).reshape((1, -1))

		prediction = self.__cache.get(prediction_input.tobytes())
		if prediction is None:
			prediction = self._transition_model(
				prediction_input
			)
			self.__cache[prediction_input.tobytes()] = prediction

		return self._prediction_to_transition_probability(initial_state, prediction, final_state)

	def _fit_model(self, X: np.ndarray, y: np.ndarray   ):
		self._transition_model.fit(X, y)

	def _update_model(self, batch=None):
		if batch is None:
			batch = self._update_batch
		self._fit_model(
			np.array(batch[0]),
			np.array(batch[1]),
		)

	def _update_transition_probability(self, initial_state: ModelBasedState, action, final_state: ModelBasedState):
		new_batch = [self._state_action_to_model_input(initial_state, action, final_state)], [
			self._get_train_output(initial_state, action, final_state)]

		if not self._enable_batch_update:
			self._update_model(new_batch)
			return

		self._update_batch[0].extend(new_batch[0])
		self._update_batch[1].extend(new_batch[1])

		if len(self._update_batch[0]) == self._update_batch_size:
			self._update_model()
			self._update_batch = ([], [])

	def get_configs(self):
		configs = super().get_configs()
		configs.update({
			"batch_update": self._enable_batch_update,
			"update_batch_size": self._update_batch_size
		})
		return configs

	def save(self, location):
		super().save(location)
		self.__transition_model.save(os.path.join(location, TRANSITION_MODEL_FILE_NAME))

	@staticmethod
	def load_configs(location):
		configs = ModelBasedAgent.load_configs(location)
		configs["model"] = keras.models.load_model(os.path.join(location, TRANSITION_MODEL_FILE_NAME))
		return configs
