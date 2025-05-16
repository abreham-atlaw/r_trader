import os
import typing
from abc import ABC, abstractmethod

import numpy as np

from lib.utils.logger import Logger


class AbstractSampleWeightMerger(ABC):

	def __init__(
			self,
			weight_paths: typing.List[str],
			export_path: str
	):
		self.__weight_paths = weight_paths
		self.__export_path = export_path

	def __setup(self):
		Logger.info(f"Setting up...")
		os.makedirs(self.__export_path, exist_ok=True)

	@abstractmethod
	def _merge(self, weights: typing.List[np.ndarray]) -> np.ndarray:
		pass

	def __export_weights(self, weights: np.ndarray, filename: str):
		np.save(os.path.join(self.__export_path, filename), weights)

	def start(self):
		self.__setup()
		filenames = sorted(os.listdir(self.__weight_paths[0]))

		for i, filename in enumerate(filenames):
			weights = [
				np.load(os.path.join(path, filename))
				for path in self.__weight_paths
			]
			merged_weights = self._merge(weights)
			self.__export_weights(merged_weights, filename)

			Logger.info(f"[+]Processed {(i + 1) * 100 / len(filenames):.2f}%", end="\r")

		Logger.info(f"Done!")
