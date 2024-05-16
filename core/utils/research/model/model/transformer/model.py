
import torch
import torch.nn as nn

from core.utils.research.model.layers.collapse_ff_block import CollapseFFBlock
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.transformer import Decoder

import matplotlib.pyplot as plt


class Transformer(nn.Module):

	def __init__(
			self,
			decoder: Decoder,
			collapse: CollapseFFBlock,
			input_size: int = 1028,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.decoder = decoder
		self.softmax = nn.Softmax(-1)
		self.collapse_block = collapse
		self.input_size = input_size
		self.extra_len = collapse.extra_len
		self.__init()

	def __init(self):
		init_data = torch.rand((1, self.input_size))
		self(init_data)

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		seq = X[:, :-self.extra_len]
		y = self.decoder(seq)
		y = self.collapse_block(y, X[:, -self.extra_len:])
		return y
