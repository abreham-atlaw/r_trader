import typing

import torch
import torch.nn as nn

import os

from core.utils.research.model.model.ensemble.stacked.stack_model import StackModel
from lib.utils.torch_utils.tensor_merger import TensorMerger
from .model_output_exporter import ModelOutputExporter


class MergedModelOutputExporter(ModelOutputExporter):

	def __init__(
			self,
			models: typing.List[nn.Module],
			*args,
			export_merger: bool = True,
			merger_export_name: str = "merger.pkl",
			**kwargs
	):
		model = StackModel(models)
		super().__init__(*args, model=model, **kwargs)
		self.merger = TensorMerger(use_shared_dims=True)
		self.__export_merger = export_merger
		self.__merger_export_path = os.path.join(self.export_path, merger_export_name)

	def _export_merger(self):
		self.merger.save_config(self.__merger_export_path)

	def _export_output(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
		merged = self.merger.merge([torch.unsqueeze(X, dim=1), y_hat])
		X_path, y_path, _ = self._generate_path()

		for array, path in zip([merged, y], [X_path, y_path]):
			self._export_array(array, path)

		if self.__export_merger:
			self._export_merger()

