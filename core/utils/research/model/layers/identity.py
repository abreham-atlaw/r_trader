import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class Identity(SpinozaModule):

	def call(self, x: torch.Tensor) -> torch.Tensor:
		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return {}
