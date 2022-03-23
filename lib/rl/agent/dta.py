from typing import *
from abc import ABC, abstractmethod

import numpy as np
from tensorflow import keras

import os
from datetime import datetime

from .mba import ModelBasedAgent
from lib.utils.logger import Logger

from temp import statics

TRANSITION_MODEL_FILE_NAME = "transition_model.h5"


class DNNTransitionAgent(ModelBasedAgent, ABC):

	def __init__(self, *args, batch_update=True, update_batch_size=100, fit_params=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._enable_batch_update = batch_update
		self._update_batch_size = update_batch_size
		self.__fit_params = fit_params
		if fit_params is None:
			self.__fit_params = {
				"epochs": 50,
				"batch_size": 4
			}
		if self._enable_batch_update:
			self._update_batch = ([], [])
			
		self.__transition_model: Union[keras.Model or None] = None
		self.__cache = {}

	@abstractmethod
	def _state_action_to_model_input(self, state, action, final_state) -> np.ndarray:
		pass

	@abstractmethod
	def _prediction_to_transition_probability(self, initial_state, output: np.ndarray, final_state) -> float:
		pass

	@abstractmethod
	def _get_train_output(self, initial_state, action, final_state) -> np.ndarray:
		pass

	@Logger.logged_method
	def _get_transition_model(self) -> keras.Model:
		if self.__transition_model is None:
			raise Exception("Transition Model not Set")
		return self.__transition_model

	def set_transition_model(self, model: keras.Model):
		self.__transition_model = model

	@Logger.logged_method
	def _get_expected_transition_probability(self, initial_state, action, final_state) -> float:
		if action is None:
			return 1

		start_time = datetime.now()
		prediction_input = self._state_action_to_model_input(initial_state, action, final_state).reshape((1, -1))
		statics.durations["state_action_to_model_input"] += (datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		prediction = self.__cache.get(prediction_input.tobytes())
		if prediction is None:
			prediction = self._get_transition_model()(
				prediction_input
			)
			self.__cache[prediction_input.tobytes()] = prediction
		else:
			statics.iterations["cached_prediction"] += 1
		statics.durations["prediction"] += (datetime.now() - start_time).total_seconds()
		statics.iterations["prediction"] += 1

		return self._prediction_to_transition_probability(initial_state, prediction, final_state)

	def _fit_model(self, X: np.ndarray, y: np.ndarray, fit_params: Dict):
		self._get_transition_model().summary()
		self._get_transition_model().fit(X, y, **fit_params)

	@Logger.logged_method
	def _update_model(self, batch=None):
		if batch is None:
			batch = self._update_batch
		self._fit_model(
			np.array(batch[0]),
			np.array(batch[1]),
			self.__fit_params
		)

	@Logger.logged_method
	def _update_transition_probability(self, initial_state, action, final_state):
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
