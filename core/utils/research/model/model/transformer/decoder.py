import typing

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

from core.utils.research.model.layers.ffn import FeedForwardNetwork
from core.utils.research.model.model.savable import SavableModule


class Decoder(SavableModule):

	def __init__(
			self,
			embedding: SavableModule,
			input_size: int,
			emb_size: int,
			num_heads: int,
			ff_size: int,
			dtype=torch.float32,
	):
		self.args = {
			"embedding": embedding,
			"emb_size": emb_size,
			"num_heads": num_heads,
			"ff_size": ff_size,
			"input_size": input_size
		}
		super().__init__()
		self.emb_size = emb_size
		self.dtype = dtype
		self.embedding = embedding
		self.pos_encoding = PositionalEncoding1D(emb_size)
		self.self_attn_layer_norm_layer = None
		self.ff_layer_norm_layer = None
		self.norm_layer = None
		self.self_attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True, dtype=dtype)
		self.ffn = FeedForwardNetwork(emb_size, ff_size, dtype=dtype)
		self.input_size = input_size
		self.__init()

	def __init(self):
		init_data = torch.rand((1, self.input_size))
		self(init_data)

	def self_attention_layer_norm(self, out: torch.Tensor) -> torch.Tensor:
		if self.ff_layer_norm_layer is None:
			self.ff_layer_norm_layer = nn.LayerNorm([out.shape[1], self.emb_size], dtype=self.dtype)
		return self.ff_layer_norm_layer(out)

	def ff_layer_norm(self, out: torch.Tensor) -> torch.Tensor:
		if self.ff_layer_norm_layer is None:
			self.ff_layer_norm_layer = nn.LayerNorm([out.shape[1], self.emb_size], dtype=self.dtype)
		return self.ff_layer_norm_layer(out)

	def norm(self, out: torch.Tensor) -> torch.Tensor:
		if self.norm_layer is None:
			self.norm_layer = nn.LayerNorm(out.shape[1])
		return self.norm_layer(out)

	def forward(self, X: torch.Tensor):
		y = self.norm(X)

		y = self.embedding(X).transpose(2, 1)

		y = y + self.pos_encoding(y).type(self.dtype)

		attn_out, attn_weights = self.self_attention(y, y, y)

		y = self.self_attention_layer_norm(y + attn_out)

		ffn_out = self.ffn(y)

		y = self.ff_layer_norm(y + ffn_out).transpose(2, 1)

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args