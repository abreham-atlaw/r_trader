import typing

import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.torch_utils.tensor_merger import TensorMerger
from .stacked_ensemble_model import StackedEnsembleModel


class SimplifiedSEM(SpinozaModule):

	def __init__(self, model: StackedEnsembleModel, merger: TensorMerger):
		super().__init__(input_size=(None, *merger.target_shape[1:]), auto_build=False)
		self.model = model
		self.merger = merger
		self.model.init()
		self.init()

	def call(self, X) -> torch.Tensor:
		X, y = self.merger.split(X)
		X = torch.squeeze(X, dim=1)
		return self.model._call(X, y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return {
			"model": self.model
		}
