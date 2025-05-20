from abc import ABC, abstractmethod

import numpy as np

import os

from lib.utils.logger import Logger


class AbstractSampleWeightManipulator(ABC):

	def __init__(
			self,
			weights_path: str,
			export_path: str,
			min_weight: float = 0.0,
	):
		self.__weights_path = weights_path
		self.__export_path = export_path
		self.__min_weight = min_weight

	def __setup(self):
		os.makedirs(self.__export_path, exist_ok=True)

	@abstractmethod
	def _manipulate(self, w: np.ndarray) -> np.ndarray:
		pass

	def start(self):
		self.__setup()

		filenames = sorted(os.listdir(self.__weights_path))

		for i, filename in enumerate(filenames):
			weights = np.load(os.path.join(self.__weights_path, filename))
			manipulated_weights = self._manipulate(weights)
			manipulated_weights[manipulated_weights < self.__min_weight] = self.__min_weight
			np.save(os.path.join(self.__export_path, filename), manipulated_weights)
			Logger.info(f"[+]Processed {(i + 1) * 100 / len(filenames):.2f}%", end="\r")

		Logger.info(f"Done!")
