import torch
from torch import nn


class OutputBatchVariance(nn.Module):
	
	def __init__(self, softmax=False):
		super(OutputBatchVariance, self).__init__()
		self.__softmax = softmax

	def forward(self, inputs):
		probabilities = inputs
		if self.__softmax:
			probabilities = torch.softmax(inputs, dim=1)

		class_variance = torch.var(probabilities, dim=0)

		mean_variance = class_variance.mean()

		return mean_variance
