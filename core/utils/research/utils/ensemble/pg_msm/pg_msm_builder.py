import hashlib
import os
import typing

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from core.utils.research.data.load import BaseDataset
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.xswg import BasicXSampleWeightGenerator
from core.utils.research.losses import SpinozaLoss
from core.utils.research.model.model.ensemble.stacked.msm.performance_grid_msm import PerformanceGridMSM
from lib.utils.logger import Logger
from core.utils.research.data.prepare.swg.standardized_swe import StandardizedSampleWeightExporter
from .performance_grid_evaluator import PerformanceGridEvaluator


class PerformanceGridMSMBuilder:

	def __init__(
			self,
			data_paths: typing.List[str],
			generators: typing.List[BasicXSampleWeightGenerator],
			tmp_path: str,
			loss: nn.Module,
			batch_size: int = 256,
			performance_grid_export_name: str = "performance_grid.npy",
			weights_std: float = 0.3,
			weights_mean: float = 1.0,
			X_dir: str = "X"
	):
		if not isinstance(loss, SpinozaLoss) or not loss.weighted_sample:
			Logger.warning(f"Detected that loss is either not SpinozaLoss or Sample Weight is not enabled.")

		self.__data_paths = data_paths
		self.__generators = generators
		self.__tmp_path = tmp_path
		self.__loss = loss
		self.__batch_size = batch_size
		self.__performance_grid_path = os.path.join(self.__tmp_path, performance_grid_export_name)
		self.__weights_std = weights_std
		self.__weights_mean = weights_mean
		self.__X_dir = X_dir

	def _generate_export_path(self, generator: AbstractSampleWeightGenerator) -> str:
		return os.path.join(self.__tmp_path, str(id(generator)))

	def __generate_weight(self, generator: AbstractSampleWeightGenerator) -> typing.Tuple[str, AbstractSampleWeightGenerator]:
		Logger.info(f"Generating weights for Generator: {generator}...")

		export_path = self._generate_export_path(generator)

		os.makedirs(export_path)
		exporter = StandardizedSampleWeightExporter(
			input_paths=list(map(lambda path: (os.path.join(path, self.__X_dir), ), self.__data_paths)),
			export_path=export_path,
			generator=generator,
			target_std=self.__weights_std,
			target_mean=self.__weights_mean
		)
		exporter.start()
		exporter.start()
		return export_path, exporter.get_standardized_generator()

	def __generate_all_weights(self) -> typing.Tuple[typing.List[str], typing.List[AbstractSampleWeightGenerator]]:
		weights_and_generators = [
			self.__generate_weight(generator)
			for generator in self.__generators
		]
		weights_paths, generators = [
			[
				weight_and_generator[i]
				for weight_and_generator in weights_and_generators
			]
			for i in range(2)
		]
		return weights_paths, generators

	def __generate_performance_grid(
			self,
			weights_path: typing.List[typing.List[str]],
			models: typing.List[str],
	) -> np.ndarray:
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
			],
		)

		grid = evaluator.evaluate(models, self.__performance_grid_path)
		return grid

	def __build_model(
			self,
			generators: typing.List[AbstractSampleWeightGenerator],
			performance_grid: np.ndarray,
			models: typing.List[nn.Module],
			*args, **kwargs
	):
		return PerformanceGridMSM(
			generators=generators,
			performance_grid=performance_grid,
			models=models,
			*args, **kwargs
		)

	def build(
			self,
			models: typing.List[nn.Module],
			*args, **kwargs
	) -> PerformanceGridMSM:

		weights_paths, generators = self.__generate_all_weights()
		performance_grid = self.__generate_performance_grid(weights_paths, models)
		model = self.__build_model(generators, performance_grid, models, *args, **kwargs)
		return model
