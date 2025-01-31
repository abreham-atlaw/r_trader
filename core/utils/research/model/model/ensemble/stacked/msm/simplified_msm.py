import typing

import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.torch_utils.tensor_merger import TensorMerger
from .masked_stacked_model import MaskedStackedModel


class SimplifiedMSM(SpinozaModule):

	def __init__(self, model: MaskedStackedModel, merger: TensorMerger):
		super().__init__(input_size=(None, *merger.target_shape), auto_build=False)
		self.model = model
		self.merger = merger
		self.init()

	def call(self, X) -> torch.Tensor:
		X, y = self.merger.split(X)
		X = torch.squeeze(X, dim=1)
		return self.model.get_and_apply_mask(X, y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return {
			"model": self.model
		}
