import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D

from core.utils.research.model.layers import Indicators, DynamicLayerNorm
from core.utils.research.model.model.savable import SpinozaModule


class EmbeddingBlock(SpinozaModule):

	def __init__(
			self,
			indicators: typing.Optional[Indicators] = None,
			positional_encoding: bool = False,
			norm_positional_encoding: bool = False,
			input_norm: nn.Module = None
	):
		self.args = {
			"indicators": indicators,
			"positional_encoding": positional_encoding,
			"norm_positional_encoding": norm_positional_encoding,
			"input_norm": input_norm
		}
		super().__init__(auto_build=False)
		self.indicators = indicators if indicators is not None else Indicators()
		self.pos_layer = None

		self.pos_norm = DynamicLayerNorm() if norm_positional_encoding else nn.Identity()
		self.pos = self.positional_encoding if positional_encoding else nn.Identity()

		self.input_norm = input_norm if input_norm is not None else nn.Identity()

	def positional_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
		if self.pos_layer is None:
			self.pos_layer = PositionalEncodingPermute1D(inputs.shape[1])
		inputs = self.pos_norm(inputs)
		return inputs + self.pos_layer(inputs)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		out = self.input_norm(x)
		out = self.indicators(out)
		out = self.pos(out)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
