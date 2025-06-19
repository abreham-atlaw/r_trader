import typing

import numpy as np
import torch
from torch import nn

from core.utils.research.data.prepare.swg.xswg import BasicXSampleWeightGenerator
from .masked_stacked_model import MaskedStackedModel


class PerformanceGridMSM(MaskedStackedModel):

	def __init__(
			self,
			generators: typing.List[BasicXSampleWeightGenerator],
			performance_grid: typing.Union[np.ndarray, torch.Tensor],
			*args,
			activation: nn.Module = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.args.update({
			"generators": generators,
			"performance_grid": performance_grid,
			"activation": activation
		})
		self.generators = generators
		self.performance_grid = torch.transpose(torch.from_numpy(performance_grid), 1, 0)
		self.activation = activation if activation is not None else nn.Identity()

	def __get_weights(self, x: torch.Tensor) -> torch.Tensor:
		weights = torch.stack([
			torch.from_numpy(generator(x.detach().numpy()))
			for generator in self.generators
		], dim=1)
		return weights

	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		w = self.__get_weights(x)
		mask = (w @ self.performance_grid)/w.shape[1]
		mask = self.activation(mask)
		return mask
