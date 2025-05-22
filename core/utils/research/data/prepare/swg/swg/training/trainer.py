import typing

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from core.utils.research.losses import SpinozaLoss, MeanSquaredErrorLoss
from lib.utils.logger import Logger
from .models import SampleWeightGenerationModel
from .dataloader import SampleWeightGeneratorDataLoader


class SampleWeightGeneratorTrainer:

	def __init__(
			self,
			model: SampleWeightGenerationModel,
			dataloader: SampleWeightGeneratorDataLoader,
			test_split: float = 0.3,
			random_seed: int = 42,
			loss_fn: SpinozaLoss = None,
			random_split: bool = True
	):

		self.__model = model
		self.__dataloader = dataloader
		self.__test_split = test_split
		self.__random_seed = random_seed
		self.__random_split = random_split

		if loss_fn is None:
			loss_fn = MeanSquaredErrorLoss()
		self.__loss_fn = loss_fn

	@staticmethod
	def __to_tensor(x) -> torch.Tensor:
		return torch.from_numpy(x).float()

	@staticmethod
	def __from_tensor(x: torch.Tensor) -> np.ndarray:
		return x.numpy()

	def __evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
		y_hat = self.__model.predict(X)
		return self.__from_tensor(self.__loss_fn(self.__to_tensor(y_hat), self.__to_tensor(y)))

	def __split_idxs(self, size) -> typing.Tuple[np.ndarray, np.ndarray]:

		if self.__random_split:
			train_idxs, test_idxs = train_test_split(np.arange(size), test_size=self.__test_split, random_state=self.__random_seed)
		else:
			train_idxs = np.arange(0, int(size * (1 - self.__test_split)))
			test_idxs = np.arange(int(size * (1 - self.__test_split)), size)
		return train_idxs, test_idxs

	def load_data(self) -> typing.Tuple[typing.Tuple[np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]]:
		X, y = self.__dataloader.load()
		train_idxs, test_idxs = self.__split_idxs(X.shape[0])

		return (X[train_idxs], y[train_idxs]), (X[test_idxs], y[test_idxs])

	def train(self) -> float:
		Logger.info("Loading data...")
		(X_train, y_train), _ = self.load_data()

		Logger.info(f"Training model(Datasize: {X_train.shape[0]})...")
		self.__model.fit(X_train, y_train)

		return self.__evaluate(X_train, y_train)

	def evaluate(self) -> float:
		_, (X_test, y_test) = self.load_data()
		return self.__evaluate(X_test, y_test)
