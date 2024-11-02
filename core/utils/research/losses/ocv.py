import torch
from torch import nn


class OutputClassesVariance(nn.Module):
	
	def __init__(self, softmax=False):
		super(OutputClassesVariance, self).__init__()
		self.__softmax = softmax

	def forward(self, inputs, *args, **kwargs):
		probabilities = inputs
		if self.__softmax:
			probabilities = torch.softmax(inputs, dim=1)

		class_variance = torch.var(probabilities, dim=1)

		mean_variance = class_variance.mean()

		return mean_variance
