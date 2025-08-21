import typing

import torch
import torch.nn as nn

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers import EncoderNoiseInjectionLayer
from core.utils.research.model.model.savable import SpinozaModule


class Lass3TransformerInputBlock(SpinozaModule):

	def __init__(
			self,
			encoder_noise_injection: EncoderNoiseInjectionLayer = None,
	):
		self.args = {
			'encoder_noise_injection': encoder_noise_injection
		}
		super().__init__()
		self.encoder_noise_injection = encoder_noise_injection

	def inject_encoder_noise(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		if self.encoder_noise_injection is None:
			return x_encoder
		return self.encoder_noise_injection(x_encoder, x_decoder)

	def call(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		x_encoder, x_decoder = x[:, 0, :], x[:, 1, :]
		x_encoder = self.inject_encoder_noise(x_encoder, x_decoder)
		return x_encoder, x_decoder

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
