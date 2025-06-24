import torch
import torch.nn as nn


class ReverseSoftmax(nn.Module):

	def __init__(self, dim=-1, min_prob: float = 1e-12):
		super().__init__()
		self.dim = dim
		self.min_prob = min_prob

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x[x == 0] = self.min_prob

		if not torch.all(x > 0):
			raise ValueError("Softmax probabilities must be strictly positive to compute log.")

		log_probs = x.log()

		logits = log_probs - log_probs.mean(dim=self.dim, keepdim=True)

		return logits
