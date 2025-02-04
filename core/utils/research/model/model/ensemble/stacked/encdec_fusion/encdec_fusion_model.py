import typing

import torch
from torch import nn

from core.utils.research.model.layers import DynamicLayerNorm
from .edf_decoder import EDFDecoder

from core.utils.research.model.model.ensemble.stacked.stacked_ensemble_model import StackedEnsembleModel
from core.utils.research.model.model.savable import SpinozaModule


class EncDecFusionModel(StackedEnsembleModel):

	def __init__(
			self,
			models: typing.List[SpinozaModule],
			encoder: nn.Module,
			decoder: typing.Union[nn.Module, EDFDecoder],
			ff: nn.Module,
			preconcat_norm: bool = False,
	):
		super().__init__(models)
		self.args.update({
			"encoder": encoder,
			"decoder": decoder,
			"ff": ff,
			"preconcat_norm": preconcat_norm
		})
		self.encoder = encoder
		self.decoder = decoder
		self.ff = ff
		self.collapse_layer = None
		self.encoder_norm, self.decoder_norm = [DynamicLayerNorm() if preconcat_norm else nn.Identity() for _ in range(2)]
		self.init()

	def collapse(self, y: torch.Tensor) -> torch.Tensor:
		if self.collapse_layer is None:
			self.collapse_layer = nn.Identity() if y.shape[-1] == self.output_size[-1] else nn.Linear(y.shape[-1], self.output_size[-1])
		return self.collapse_layer(y)

	def _call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(x)
		decoded = self.decoder(y)

		encoded = self.encoder_norm(encoded)
		decoded = self.decoder_norm(decoded)

		concatenated = torch.cat([encoded, decoded], dim=1)

		out = self.ff(concatenated)

		out = self.collapse(out)
		return out
