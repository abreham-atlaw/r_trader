import torch

from core.utils.research.losses.spinoza_loss import SpinozaLoss


class PredictionConfidenceScore(SpinozaLoss):
	def __init__(self, *args, softmax=False, **kwargs):
		super(PredictionConfidenceScore, self).__init__(*args, **kwargs)
		self.__softmax = softmax

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		probabilities = y_hat
		if self.__softmax:
			probabilities = torch.softmax(y_hat, dim=1)

		max_confidences, _ = torch.max(probabilities, dim=1)

		avg_confidence_score = max_confidences
		return avg_confidence_score
