import typing

import torch.nn as nn


class LinearModel(nn.Module):
	def __init__(self, block_size: int, vocab_size: int, layer_sizes: typing.List[int], dropout_rate: float, activation='relu', init='kaiming'):
		super(LinearModel, self).__init__()
		self.layers = nn.ModuleList()
		layer_sizes = [block_size] + layer_sizes + [vocab_size]
		for i in range(len(layer_sizes) - 1):
			self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
			if init == 'kaiming':
				nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity=activation)
			elif init == 'xavier':
				nn.init.xavier_uniform_(self.layers[-1].weight)
			else:
				raise ValueError("Invalid initialization function. Valid initializations are 'kaiming' and 'xavier'.")
		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU()
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		else:
			raise ValueError("Invalid activation function. Valid activations are 'relu', 'leaky_relu' and 'tanh'.")
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
