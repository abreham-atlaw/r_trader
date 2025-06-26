import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute1D
from torch import nn


class PositionalEncoding(nn.Module):

	def __init__(self):
		super(PositionalEncoding, self).__init__()
		self.pos_layer = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.pos_layer is None:
			self.pos_layer = PositionalEncodingPermute1D(x.shape[1])
		return x + self.pos_layer(x)
