import torch

from core.utils.research.losses.spinoza_loss import SpinozaLoss


class OutputBatchVarianceScore(SpinozaLoss):
	
	def __init__(self, softmax=False):
		super(OutputBatchVarianceScore, self).__init__()
		self.__softmax = softmax

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		probabilities = y_hat
		if self.__softmax:
			probabilities = torch.softmax(y_hat, dim=1)

		class_variance = torch.var(probabilities, dim=0)

		mean_variance = class_variance

		return mean_variance
