
import torch
import torch.nn as nn

from core.utils.research.model.model.decoder import Decoder


class Transformer(nn.Module):

	def __init__(self, decoder: Decoder, vocab_size: int, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.decoder = decoder
		self.linear = nn.Linear(decoder.emb_size, vocab_size)
		self.softmax = nn.Softmax(-1)
		self.norm = nn.LayerNorm(decoder.block_size, )

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		X = self.norm(X)
		y = self.decoder(X)
		y = y[:, -1]
		y = self.linear(y)
		y = self.softmax(y)
		return y
