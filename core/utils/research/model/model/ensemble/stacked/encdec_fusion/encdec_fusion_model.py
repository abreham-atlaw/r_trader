import typing

import torch
from torch import nn

from core.utils.research.model.layers import DynamicLayerNorm
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock

from core.utils.research.model.model.ensemble.stacked.stacked_ensemble_model import StackedEnsembleModel
from core.utils.research.model.model.savable import SpinozaModule


class EncDecFusionModel(StackedEnsembleModel):

	def __init__(
			self,
			models: typing.List[SpinozaModule],
			encoder: nn.Module,
			decoder: nn.Module,
			collapse_block: CollapseBlock,
			channel_block: nn.Module = None,
			preconcat_norm: bool = False,
	):
		super().__init__(models)
		self.args.update({
			"encoder": encoder,
			"decoder": decoder,
			"channel_block": channel_block,
			"collapse_block": collapse_block,
			"preconcat_norm": preconcat_norm
		})

		self.encoder = encoder
		self.decoder = decoder
		self.encoder_reshape_layer = None

		self.channel_block = channel_block if channel_block is not None else nn.Identity()
		self.ff_block = collapse_block
		self.collapse_layer = None
		self.encoder_norm, self.decoder_norm = [DynamicLayerNorm() if preconcat_norm else nn.Identity() for _ in range(2)]
		self.init()

	def collapse(self, y: torch.Tensor) -> torch.Tensor:
		if self.collapse_layer is None:
			self.collapse_layer = nn.Identity() if y.shape[-1] == self.output_size[-1] else nn.Linear(y.shape[-1], self.output_size[-1])
		return self.collapse_layer(y)

	def encoder_reshape(self, encoded: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
		if self.encoder_reshape_layer is None:
			self.encoder_reshape_layer = nn.Identity() if encoded.shape[2] == decoded.shape[2] else nn.Linear(encoded.shape[2], decoded.shape[2])
		return self.encoder_reshape_layer(encoded)

	def _call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(x)
		decoded = self.decoder(y)
		encoded = self.encoder_reshape(encoded, decoded)

		encoded = self.encoder_norm(encoded)
		decoded = self.decoder_norm(decoded)

		concatenated = torch.cat([encoded, decoded], dim=1)

		channeled = self.channel_block(concatenated)

		out = self.ff_block(channeled)

		out = self.collapse(out)
		return out
