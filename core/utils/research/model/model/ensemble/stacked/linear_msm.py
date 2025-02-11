import typing

import torch
from torch import nn

from core.utils.research.model.model.linear.model import LinearModel
from .masked_stacked_model import MaskedStackedModel


class LinearMSM(MaskedStackedModel):

	def __init__(
			self,
			*args,
			ff: typing.Optional[nn.Module] = None,
			bias: bool = False,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.args.update({
			"ff": ff,
			"bias": bias
		})
		if ff is None:
			ff = nn.Identity()

		self.ff = ff
		self.collapse_layer = None
		self.bias = bias
		self.init()

	def collapse(self, x, y):
		if self.collapse_layer is None:
			self.collapse_layer = nn.Linear(x.shape[1], y.shape[1], bias=self.bias)
		return self.collapse_layer(x)

	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		out = self.ff(x)
		out = self.collapse(out, y)
		return out
