import typing

import torch
from torch import nn


class StackModel(nn.Module):

	def __init__(self, models: typing.List[nn.Module]):
		super().__init__()
		self.models = nn.ModuleList(models)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			y = torch.stack([
				m(x).detach() for m in self.models
			], dim=1)
		return y