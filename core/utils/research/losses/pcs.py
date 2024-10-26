import torch
import torch.nn as nn


class PredictionConfidenceScore(nn.Module):
	def __init__(self, softmax=False):
		super(PredictionConfidenceScore, self).__init__()
		self.__softmax = softmax

	def forward(self, inputs, *args, **kwargs):
		probabilities = inputs
		if self.__softmax:
			probabilities = torch.softmax(inputs, dim=1)

		max_confidences, _ = torch.max(probabilities, dim=1)

		avg_confidence_score = max_confidences.mean()
		return avg_confidence_score
