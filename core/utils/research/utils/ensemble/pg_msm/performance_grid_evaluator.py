import typing

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from core.utils.research.losses import SpinozaLoss
from core.utils.research.utils.model_evaluator import ModelEvaluator
from lib.utils.logger import Logger


class PerformanceGridEvaluator:

	def __init__(
			self,
			dataloaders: typing.List[DataLoader],
			loss: SpinozaLoss
	):
		self.__dataloaders = dataloaders
		self.__loss = loss
		self.__evaluators = self.__init_evaluators()

	def __init_evaluators(self):
		return [
			ModelEvaluator(
				cls_loss_fn=self.__loss,
				dataloader=dataloader,
				reg_loss_fn=None
			)

			for dataloader in self.__dataloaders
		]

	@staticmethod
	def __evaluate_model(model, evaluator: ModelEvaluator) -> float:
		loss, _, __ = evaluator(model)
		return 1/loss

	@staticmethod
	def __export_output(performance, export_path):
		Logger.info(f"Saving performance to {export_path}")
		np.save(export_path, performance)

	def evaluate(
		self,
		models: typing.List[nn.Module],
		export_path: str = None
	) -> np.ndarray:

		performance = np.zeros((len(models), len(self.__evaluators)))

		for i, model in enumerate(models):
			for j, evaluator in enumerate(self.__evaluators):
				Logger.info(f"Evaluating model {i+1}/{len(models)} on data {j+1}/{len(self.__evaluators)}...")
				performance[i, j] = self.__evaluate_model(model, evaluator)

		Logger.success(f"Evaluation complete!")

		if export_path is not None:
			self.__export_output(performance, export_path)

		return performance
