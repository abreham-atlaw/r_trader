import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class Lass2DModel(SpinozaModule):

	def __init__(self, model: SpinozaModule, look_back: int = None):
		self.args = {
			"model": model,
			"look_back": look_back
		}
		if look_back is None:
			look_back = model.input_size[-1]
		super().__init__(input_size=model.input_size[-1]+look_back, auto_build=False)
		self.look_back = look_back
		self.model = model
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:

		inputs = torch.repeat_interleave(torch.unsqueeze(x[..., :self.model.input_size[-1]], dim=1), self.model.input_size[1], dim=1)
		inputs[:, 1, -1] = 0
		for i in range(self.look_back):
			prediction = self.model(inputs)
			inputs[:, 1, -1] = torch.squeeze(prediction)
			inputs[:, :, 0:-1] = inputs[:, :, 1:]
			inputs[:, 0, -1] = x[:, -self.model.input_size[-1]+i]
			inputs[:, 1, -1] = 0

		return self.model(inputs)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
