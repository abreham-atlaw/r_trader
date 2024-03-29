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

	def __init__(self, *args, batch_update=True, update_batch_size=100,  clear_update_batch=True, **kwargs):
		super().__init__(*args, **kwargs)
		self._enable_batch_update = batch_update
		self._update_batch_size = update_batch_size
		if self._enable_batch_update:
			self._update_batch = ([], [])
		self._clear_update_batch = clear_update_batch

		self.__transition_model: Union[Model or None] = None
		self.__cache = {}
		self._validation_dataset = ([], [])

	@abstractmethod
	def _prepare_dta_input(self, state: List[ModelBasedState], action: List[Any], final_state: List[ModelBasedState]) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_dta_output(self, initial_state: List[ModelBasedState], output: np.ndarray, final_state: List[ModelBasedState]) -> List[float]:
		pass

	@abstractmethod
	def _prepare_dta_train_output(self, initial_state: List[ModelBasedState], action: List[Any], final_state: List[ModelBasedState]) -> np.ndarray:
		pass

	@property
	def _transition_model(self) -> Model:
		if self.__transition_model is None:
			raise Exception("Transition Model not Set")
		return self.__transition_model

	def set_transition_model(self, model: Model):
		self.__transition_model = model

	def __get_cached(self, inputs: np.ndarray) -> np.ndarray:
		out = np.array([np.nan for _ in range(len(inputs))])
		for i in range(len(inputs)):
			cached = self.__cache.get(inputs[i].tobytes())
			if cached is not None:
				out[i] = cached
		return out

	def __cache_predictions(self, inputs: np.ndarray, predictions: np.ndarray):
		for i in range(len(inputs)):
			self.__cache[inputs[i].tobytes()] = predictions[i]

	def _predict(self, model: Model, inputs: np.array) -> np.ndarray:
		return model.predict(inputs)

	def _get_expected_transition_probability_distribution(self, initial_states: List[ModelBasedState], action: List[Any], final_states: List[ModelBasedState]) -> List[float]:
		prediction_input = self._prepare_dta_input(initial_states, action, final_states)

		prediction = self.__get_cached(prediction_input)

		not_cached_indexes = np.isnan(prediction)
		if np.any(not_cached_indexes):
			model_inputs, indices = np.unique(prediction_input[not_cached_indexes], axis=0, return_inverse=True)
			model_predictions = self._predict(self._transition_model, model_inputs)
			model_predictions = model_predictions[indices]
			prediction[not_cached_indexes] = self._prepare_dta_output(
				initial_states,
				model_predictions,
				final_states
			)
			self.__cache_predictions(prediction_input[not_cached_indexes], prediction[not_cached_indexes])

		return list(prediction)

	def _fit_model(self, X: np.ndarray, y: np.ndarray):
		self._transition_model.fit(X, y)

	def _evaluate_model(self, X: np.ndarray, y: np.ndarray):
		self._transition_model.evaluate(X, y)

	def _update_model(self, batch=None):
		if batch is None:
			batch = self._update_batch
		self._fit_model(
			np.array(batch[0]),
			np.array(batch[1]),
			self.__fit_params
		)
		if len(self._validation_dataset[0]) == 0:
			return
		self._evaluate_model(
			np.array(self._validation_dataset[0]),
			np.array(self._validation_dataset[1])
		)

	def __add_validation_set(self, X, y):
		self._validation_dataset[0].extend(X)
		self._validation_dataset[1].extend(y)

	def _update_transition_probability(self, initial_state: ModelBasedState, action, final_state: ModelBasedState):
		new_batch = (
			self._prepare_dta_input([initial_state], [action], [final_state]),
			self._prepare_dta_train_output([initial_state], [action], [final_state])
		)

		if not self._enable_batch_update:
			self._update_model(new_batch)
			return

		self._update_batch[0].extend(new_batch[0])
		self._update_batch[1].extend(new_batch[1])

		if len(self._update_batch[0]) >= self._update_batch_size:
			self._update_model()
			self.__add_validation_set(*self._update_batch)
			if self._clear_update_batch:
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
