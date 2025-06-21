import typing
from abc import ABC

import numpy as np

import os

from lib.utils.logger import Logger
from .abstract_swg import AbstractSampleWeightGenerator


class SampleWeightExporter(ABC):

	def __init__(
			self,
			input_paths: typing.List[typing.Tuple[str, ...]],
			export_path: str,
			generator: AbstractSampleWeightGenerator
	):

		if isinstance(input_paths[0], str):
			input_paths = [input_paths]

		self.__input_paths = input_paths
		self.__export_path = export_path
		self._generator = generator

	def __setup(self):
		Logger.info(f"Setting up...")
		os.makedirs(self.__export_path, exist_ok=True)

	def __export_weights(self, weights: np.ndarray, filename: str):
		np.save(os.path.join(self.__export_path, filename), weights)

	def __process_path_batch(self, input_paths: typing.Tuple[str, ...]):
		Logger.info(f"Processing {input_paths}")

		filenames = os.listdir(input_paths[0])

		for i, filename in enumerate(filenames):
			inputs = [
				np.load(os.path.join(input_path, filename))
				for input_path in input_paths
			]
			weights = self._generator(*inputs)
			self.__export_weights(weights, filename)

			Logger.info(f"[+]Processed {(i + 1) * 100 / len(filenames):.2f}%", end="\r")
		Logger.success("Successfully processed files!")

	def start(self):
		self.__setup()
		Logger.info(f'Processing {len(self.__input_paths)} directories...')
		for input_paths in self.__input_paths:
			self.__process_path_batch(input_paths)
		Logger.success(f"Done!")
