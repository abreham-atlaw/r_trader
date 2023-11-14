import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

from core.utils.research.model.layers.ffn import FeedForwardNetwork


class Decoder(nn.Module):

	def __init__(
			self,
			kernel_size: int,
			emb_size: int,
			block_size: int,
			num_heads: int,
			ff_size: int,
			dtype=torch.float32
	):
		super().__init__()
		self.emb_size = emb_size
		self.block_size = block_size
		self.dtype = dtype
		self.conv = nn.Sequential(
			nn.Conv1d(1, emb_size, kernel_size, padding="same"),
			nn.Tanh(),
			nn.Conv1d(emb_size, emb_size, kernel_size, padding="same"),
			nn.Tanh(),
		)
		self.pos_encoding = PositionalEncoding1D(emb_size)
		self.self_attn_layer_norm = nn.LayerNorm([block_size, emb_size], dtype=dtype)
		self.ff_layer_norm = nn.LayerNorm([block_size, emb_size], dtype=dtype)
		self.self_attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True, dtype=dtype)
		self.ffn = FeedForwardNetwork(emb_size, ff_size, dtype=dtype)

	def forward(self, X: torch.Tensor):
		y = self.conv(X.unsqueeze(1)).transpose(2, 1)

		y = y + self.pos_encoding(y).type(self.dtype)

		attn_out, attn_weights = self.self_attention(y, y, y)

		y = self.self_attn_layer_norm(y + attn_out)

		ffn_out = self.ffn(y)

		y = self.ff_layer_norm(y + ffn_out)

		return y
