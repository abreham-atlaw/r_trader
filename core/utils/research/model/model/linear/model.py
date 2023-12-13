import typing

import torch
import torch.nn as nn


class LinearModel(nn.Module):
	def __init__(self, block_size: int, vocab_size: int, layer_sizes: typing.List[int], dropout_rate: float, leaky_relu=True):
		super(LinearModel, self).__init__()
		self.layers = nn.ModuleList()
		layer_sizes = [block_size] + layer_sizes + [vocab_size]
		for i in range(len(layer_sizes) - 1):
			self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			if not leaky_relu:
				nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity='relu')
		self.activation = nn.ReLU()
		if leaky_relu:
			self.activation = nn.LeakyReLU()
		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
			x = self.activation(x)
			x = self.dropout(x)
		return x
