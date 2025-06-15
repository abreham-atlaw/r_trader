import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class LayerStack(SpinozaModule):

	def __init__(self, layers: typing.List[nn.Module], dim=1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.args = {
			"layers": layers,
			"dim": dim
		}
		self.layers = nn.ModuleList(layers)
		self.dim = dim

	def call(self, x: torch.Tensor) -> torch.Tensor:
		assert x.shape[self.dim] == len(self.layers)
		split_inputs = torch.unbind(x, dim=self.dim)
		outputs = [layer(input) for input, layer in zip(split_inputs, self.layers)]
		return torch.stack(outputs, dim=self.dim)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
