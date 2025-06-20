import os
import typing

import numpy as np

from core.utils.research.data.prepare.swg.abstract_swe import SampleWeightExporter
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.manipulate import StandardizeSampleWeightManipulator
from core.utils.research.data.prepare.swg.swg_pipeline import SampleWeightGeneratorPipeline
from lib.utils.logger import Logger


class PerformanceGridSampleWeightExporter(SampleWeightExporter):

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
		mean, std = self.__get_standardization_configs()
		Logger.info(f"Using Mean: {mean}, Using STD: {std}")
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
		return mean, std

	def __create_generator(self, mean, std) -> AbstractSampleWeightGenerator:
		return SampleWeightGeneratorPipeline(
			generators=[

			]
		)

	def start(self):
		super().start()
		mean, std = self.__standardize_weights()
