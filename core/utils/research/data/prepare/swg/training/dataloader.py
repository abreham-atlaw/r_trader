import typing

import numpy as np
from torch.utils.data import DataLoader

from lib.utils.logger import Logger


class SampleWeightGeneratorDataLoader:

	def __init__(
			self,
			dataloader: DataLoader,
			bounds: typing.List[float],
			X_extra_len: int = 124,
			y_extra_len: int = 1
	):
		self.__dataloader = dataloader
		self.__X_extra_len = X_extra_len
		self.__y_extra_len = y_extra_len
		if isinstance(bounds, list):
			bounds = np.array(bounds)
		self.__bounds = bounds

	def __merge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		X, y = [X[:, :-self.__X_extra_len], y[:, :-self.__y_extra_len]]
		y = X[:, -1] * self.__bounds[np.argmax(y, axis=1)]
		return np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

	def load(self) -> typing.Tuple[np.ndarray, np.ndarray]:
		X, y = None, None

		for i, (X_batch, y_batch, w_batch) in enumerate(self.__dataloader):

			X_batch, y_batch, w_batch = [arr.numpy() for arr in [X_batch, y_batch, w_batch]]

			X_batch = self.__merge(X_batch, y_batch)

			if X is None:
				X = X_batch
				y = w_batch
				continue

			X = np.concatenate((X, X_batch), axis=0)
			y = np.concatenate((y, w_batch), axis=0)

			Logger.info(f"[+]Loaded {(i+1)*100/len(self.__dataloader) :.2f}%...", end="\r")

		return X, y


