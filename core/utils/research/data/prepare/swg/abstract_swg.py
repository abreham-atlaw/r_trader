import os
from abc import ABC, abstractmethod

import numpy as np

from lib.utils.logger import Logger


class AbstractSampleWeightGenerator(ABC):

	def __init__(
			self,
			data_path: str,
			export_path: str,
			X_dir: str = "X",
			y_dir: str = "y"
	):
		self.__data_path = data_path
		self.__export_path = export_path
		self.__X_dir = X_dir
		self.__y_dir = y_dir

	def __setup(self):
		Logger.info(f"Setting up...")
		os.makedirs(self.__export_path, exist_ok=True)

	@abstractmethod
	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		pass

	def __export_weights(self, weights: np.ndarray, filename: str):
		np.save(os.path.join(self.__export_path, filename), weights)

	def start(self):
		self.__setup()
		filenames = os.listdir(os.path.join(self.__data_path, self.__X_dir))

		for i, filename in enumerate(filenames):
			X, y = [
				np.load(os.path.join(self.__data_path, dir_name, filename))
				for dir_name in [self.__X_dir, self.__y_dir]
			]
			weights = self._generate_weights(X, y)
			self.__export_weights(weights, filename)

			Logger.info(f"[+]Processed {(i + 1) * 100 / len(filenames):.2f}%", end="\r")

		Logger.info(f"Done!")
