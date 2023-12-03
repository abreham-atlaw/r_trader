import torch
import torch.nn as nn


class CNN(nn.Module):

	def __init__(self, num_classes, conv_channels, kernel_sizes, pool_sizes, dropout_rate, kaiming_normal=False):
		super(CNN, self).__init__()
		self.layers = nn.ModuleList()
		for i in range(len(conv_channels) - 1):
			layers = [
				nn.Conv1d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=kernel_sizes[i], stride=1, padding=1),
				nn.ReLU()
			]
			if kaiming_normal:
				nn.init.kaiming_normal_(layers[0].weight, mode='fan_out', nonlinearity='relu')
			if pool_sizes[i] > 0:
				layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i], stride=2))
			if dropout_rate > 0:
				layers.append(nn.Dropout(dropout_rate))
			self.layers.append(nn.Sequential(*layers))
		self.avg_pool = nn.AdaptiveAvgPool1d((1,))
		self.fc = nn.Linear(conv_channels[-1], num_classes)
		if kaiming_normal:
			nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
		if dropout_rate > 0:
			self.dropout = nn.Dropout(dropout_rate)
		else:
			self.dropout = nn.Identity()

	def forward(self, x):
		out = torch.unsqueeze(x, dim=1)
		for layer in self.layers:
			out = layer(out)
		out = self.avg_pool(out)
		out = out.reshape(out.size(0), -1)
		out = self.dropout(out)
		out = self.fc(out)
		return out
