import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule

from .masked_stacked_model import MaskedStackedModel


class OutputAwareLMSM(MaskedStackedModel):

	def __init__(
			self,
			encoder: nn.Module,
			decoder: nn.Module,
			ff: nn.Module,
			*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.args.update({
			"encoder": encoder,
			"decoder": decoder,
			"ff": ff
		})
		self.encoder = encoder
		self.decoder = decoder
		self.ff = ff
		self.collapse_layer = None
		self.init()

	def collapse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if self.collapse_layer is None:
			self.collapse_layer = nn.Conv1d(x.shape[1], y.shape[1], padding=1)
		return self.collapse_layer(x)

	def _get_mask(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(x)
		decoded = self.decoder(y)

		out = torch.concatenate((encoded, decoded), dim=1)
		out = self.ff(out)
		out = self.collapse(out, y)
		return out
