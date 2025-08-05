import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger


class LassHorizonModel(SpinozaModule):

	def __init__(
			self,
			h: float,
			model: SpinozaModule,
			max_depth: int = None
	):
		super().__init__(input_size=model.input_size, output_size=model.output_size, auto_build=False)
		self.args = {
			"h": h,
			"model": model
		}
		if max_depth is None:
			max_depth = model.input_size[-1] // 2

		Logger.info(f"Initializing LassHorizonModel(h={h}, max_depth={max_depth})...")
		self.h = h
		self.model = model
		self.max_depth = max_depth

	def set_h(self, h: float):
		self.h = h

	def check_max_depth(self, x: torch.Tensor) -> bool:
		return torch.any(x[:, :, :self.max_depth] != 0)

	def shift_and_predict(self, x: torch.Tensor) -> torch.Tensor:
		x[:, :, 1:] = x[:, :, :-1].clone()
		x[:, :, 0] = 0
		y = self(x)
		return torch.squeeze(y)

	def process_sample(self, x: torch.Tensor) -> torch.Tensor:
		x[:, 1, -1] = self.shift_and_predict(x.clone())
		return x

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = x.clone()
		sample_mask = torch.rand(x.size(0)) <= self.h

		if self.check_max_depth(x) and torch.any(sample_mask):
			x[sample_mask] = self.process_sample(x[sample_mask])

		return self.model(x)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		Logger.warning("Exporting HorizonModel. This is not recommended as it should be used as a wrapper. Use HorizonModel.model instead.")
		return self.args
