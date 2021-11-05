from typing import *
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from tensorflow.python import keras

from sklearn.model_selection import train_test_split


class KerasTrainer(ABC):

	def __init__(self,
				data_path,
				export_path,
				epochs: int = 20,
				batch_size: int = 32,
				test: bool = True,
				test_split_ratio: float = 0.3,
				fit_params: dict = None):

		self._data_path = data_path
		self._export_path = export_path
		self._fit_params = fit_params
		if fit_params is None:
			self._fit_params = {}
		self._epochs = epochs
		self._batch_size = batch_size
		self._test = test
		self._test_split_ratio = test_split_ratio

		if fit_params is None:
			self.fit_params = {}

	def _load_data(self, data_path) -> pd.DataFrame:
		print(f"[+]Loading Data: {data_path}")
		return pd.read_csv(data_path)

	@abstractmethod
	def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
		pass

	def _split_data(self, X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		print("[+]Splitting Data...")
		return train_test_split(X, y, test_size=self._test_split_ratio, random_state=42)

	@abstractmethod
	def _create_model(self) -> keras.Model:
		pass

	def _compile_model(self, model: keras.Model):
		model.compile()

	def _save_model(self, model: keras.Model, save_path):
		print(f"[+]Saving Model to {save_path}")
		model.save(save_path)

	def start(self):
		print("[+]Starting Training...")
		raw_data = self._load_data(self._data_path)
		X, y = self._prepare_data(raw_data)
		if self._test:
			X_train, X_test, y_train, y_test = self._split_data(X, y)
		else:
			X_trian, y_train = X, y
		del X, y
		model: keras.Model = self._create_model()
		self._compile_model(model)
		print("[+]Model Summary")
		print(model.summary())
		history = model.fit(
			X_train,
			y_train,
			epochs=self._epochs,
			batch_size=self._batch_size,
			**self._fit_params
		)
		del X_train, y_train
		test_history = None
		if self._test:
			test_history = model.evaluate(X_test, y_test)

		self._save_model(model, self._export_path)

		return history, test_history
