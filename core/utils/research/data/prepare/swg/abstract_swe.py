import typing
from abc import ABC

import numpy as np

import os

from lib.utils.logger import Logger
from .abstract_swg import AbstractSampleWeightGenerator


class SampleWeightExporter(ABC):

	def __init__(
			self,
			input_paths: typing.List[str],
			export_path: str,
			generator: AbstractSampleWeightGenerator
	):
		self.__input_paths = input_paths
		self.__export_path = export_path
		self.__generator = generator

	def __setup(self):
		Logger.info(f"Setting up...")
		os.makedirs(self.__export_path, exist_ok=True)

	def __export_weights(self, weights: np.ndarray, filename: str):
		np.save(os.path.join(self.__export_path, filename), weights)

	def start(self):
		self.__setup()
		filenames = os.listdir(self.__input_paths[0])

		for i, filename in enumerate(filenames):
			inputs = [
				np.load(os.path.join(input_path, filename))
				for input_path in self.__input_paths
			]
			weights = self.__generator(*inputs)
			self.__export_weights(weights, filename)

			Logger.info(f"[+]Processed {(i + 1) * 100 / len(filenames):.2f}%", end="\r")

		Logger.info(f"Done!")
