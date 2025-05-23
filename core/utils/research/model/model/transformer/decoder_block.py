import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer.teb import TransformerEmbeddingBlock


class DecoderBlock(SpinozaModule):

	def __init__(
			self,
			transformer_embedding_block: TransformerEmbeddingBlock,
			num_heads: int,
			embedding_last: bool = False
	):
		self.args = {
			'transformer_embedding_block': transformer_embedding_block,
			'num_heads': num_heads,
			"embedding_last": embedding_last
		}
		super().__init__()
		self.embedding_block = transformer_embedding_block
		self.self_attention_layer = None
		self.num_heads = num_heads
		self.embedding_last = embedding_last

	def self_attention(self, x: torch.Tensor) -> torch.Tensor:
		if self.self_attention_layer is None:
			self.self_attention_layer = nn.MultiheadAttention(
				embed_dim=x.shape[-1],
				num_heads=self.num_heads,
				batch_first=True,
			)
		out, weights = self.self_attention_layer(x, x, x)
		return out

	def call(self, x) -> torch.Tensor:
		embedded = self.embedding_block(x)
		out = self.self_attention(embedded)
		if self.embedding_last:
			out = torch.transpose(out, 1, 2)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
