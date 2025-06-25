import os
import typing

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline
from lib.utils.logger import Logger


class StandardizedSampleWeightExporter(SampleWeightExporter):

	def __init__(
		self,
		input_paths: typing.List[str],
		export_path: str,
		generator: AbstractSampleWeightGenerator,
		target_mean: float = 1.0,
		target_std: float = 0.3
	):
		super().__init__(input_paths, export_path, generator)
		self.__export_path = export_path
		self.__target_mean, self.__target_std = target_mean, target_std
		self.__standardized_generator = None

	def __load_weights(self) -> np.ndarray:
		filenames = sorted(os.listdir(self.__export_path))
		return np.concatenate([
			np.load(os.path.join(self.__export_path, filename))
			for filename in filenames
		])

	def __get_standardization_configs(self) -> typing.Tuple[float, float]:
		w = self.__load_weights()
		mean, std = np.mean(w), np.std(w)
		return mean, std

	def __standardize_weights(self):
		Logger.info("Standardizing Weights...")
		mean, std = self.__get_standardization_configs()
		Logger.info(f"Target mean: {self.__target_mean}, std: {self.__target_std}")
		Logger.info(f"Current mean: {mean}, std: {std}")
		generator = StandardizeSampleWeightManipulator(
			target_std=self.__target_std,
			target_mean=self.__target_std,
			current_std=std,
			current_mean=mean
		)
		exporter = SampleWeightExporter(
			input_paths=[self.__export_path],
			export_path=self.__export_path,
			generator=generator
		)
		exporter.start()
		self.__standardized_generator = self.__create_generator(mean, std)

	def __create_generator(self, mean, std) -> AbstractSampleWeightGenerator:
		Logger.info(f"Creating Standardized Generator with mean={mean}, std={std}")
		return SampleWeightGeneratorPipeline(
			generators=[
				StandardizeSampleWeightManipulator(
					target_std=self.__target_std,
					target_mean=self.__target_mean,
					current_std=std,
					current_mean=mean
				),
				self._generator
			]
		)

	def get_standardized_generator(self) -> AbstractSampleWeightGenerator:
		if self.__standardized_generator is None:
			raise ValueError("Standardized Generator not created yet. Generation must be complete before calling get_standardized_generator.")
		return self.__standardized_generator

	def start(self):
		super().start()
		self.__standardize_weights()
