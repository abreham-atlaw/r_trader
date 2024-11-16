import torch
from torch import nn

from lib.utils.cache import Cache


class CachedModel(nn.Module):

	def __init__(self, model: nn.Module, cache_size=1000):
		super().__init__()
		self.model = model
		self.cache = Cache(cache_size=cache_size)

	@staticmethod
	def _get_key(inputs: torch.Tensor):
		return inputs.detach().numpy().tobytes()

	def forward(self, inputs: torch.Tensor):
		return self.cache.cached_or_execute(
			self._get_key(inputs),
			lambda: self.model(inputs)
		)

