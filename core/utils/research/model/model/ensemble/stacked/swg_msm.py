import typing


import torch

from core.utils.research.data.prepare.swg import AbstractSampleWeightGenerator
from .masked_stacked_model import MaskedStackedModel


class SampleWeightGeneratorMSM(MaskedStackedModel):

	def __init__(
			self,
			*args,
			swgs: typing.List[AbstractSampleWeightGenerator],
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__swgs = swgs

	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

		torch.stack([
			swg(x, y)
			for swg in self.__swgs
		])

