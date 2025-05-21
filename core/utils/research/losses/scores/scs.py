import torch
from core.utils.research.losses.spinoza_loss import SpinozaLoss


class SoftConfidenceScore(SpinozaLoss):

	def __init__(self, *args, softmax=False, temperature: float = 1e-2, **kwargs):
		super(SoftConfidenceScore, self).__init__(*args, **kwargs)
		self.__softmax = softmax
		self.temperature = temperature

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		if self.__softmax:
			y_hat = torch.softmax(y_hat, dim=1)

		weights = torch.softmax(y_hat / self.temperature, dim=1)

		soft_confidence = torch.sum(y_hat * weights, dim=1)

		return soft_confidence
