import hashlib
import os
import typing

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from core.utils.research.data.load import BaseDataset
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.model.model.ensemble.stacked.msm.performance_grid_msm import PerformanceGridMSM
from lib.utils.logger import Logger
from .pg_swe import PerformanceGridSampleWeightExporter
from .performance_grid_evaluator import PerformanceGridEvaluator


class PerformanceGridMSMBuilder:

	def __init__(
			self,
			data_paths: typing.List[str],
			generators: typing.List[AbstractSampleWeightGenerator],
			tmp_path: str,
			loss: nn.Module,
			batch_size: int = 256,
			performance_grid_export_name = "performance_grid.npy"
	):
		self.__data_paths = data_paths
		self.__generators = generators
		self.__tmp_path = tmp_path
		self.__loss = loss
		self.__batch_size = batch_size
		self.__performance_grid_path = os.path.join(self.__tmp_path, performance_grid_export_name)

	def _generate_export_path(self, data_path: str, generator: AbstractSampleWeightGenerator) -> str:
		return os.path.join(self.__tmp_path, str(id(generator)), hashlib.md5(data_path).hexdigest())

	def __generate_weight(self, generator: AbstractSampleWeightGenerator) -> typing.List[str]:
		Logger.info(f"Generating weights for Generator: {generator}...")

		paths = []

		for data_path in self.__data_paths:
			export_path = self._generate_export_path(generator, data_path)

			Logger.info(f"Generating weights for path: {data_path}...")

			if os.path.exists(export_path):
				Logger.info(f"{generator} on {data_path} already generated!")
				return export_path
			os.mkdir(export_path)
			exporter = PerformanceGridSampleWeightExporter(
				input_paths=data_path,
				export_path=export_path,
				generator=generator
			)
			exporter.start()
			paths.append(export_path)

		return paths

	def __generate_all_weights(self) -> typing.List[typing.List[str]]:
		return [
			self.__generate_weight(generator)
			for generator in self.__generators
		]

	def __generate_performance_grid(self, weights_path: typing.List[typing.List[str]], models: typing.List[str]) -> np.ndarray:
		evaluator = PerformanceGridEvaluator(
			loss=self.__loss,
			dataloaders=[
				DataLoader(
					dataset=BaseDataset(
						root_dirs=self.__data_paths,
						w_dir=os.path.abspath(path),
						load_weights=True,
					),
					batch_size=self.__batch_size
				)
				for path in weights_path
			]
		)

		grid = evaluator.evaluate(models, self.__performance_grid_path)
		return grid

	def build(self, models: typing.List[nn.Module]) -> PerformanceGridMSM:

		weights_path = self.__generate_all_weights()









