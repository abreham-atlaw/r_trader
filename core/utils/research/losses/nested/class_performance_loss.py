import torch
import torch.nn as nn


class ClassPerformanceLoss(nn.Module):

	def __init__(
			self,
			loss_fn: nn.Module,
			n: int,
			epsilon=1e-9,
			nan_to=torch.nan,
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.loss_fn = loss_fn
		self.n = n
		self.epsilon = epsilon
		self.nan_to = nan_to

	def __get_cls_loss(self, cls, loss, y_cls):
		mask = y_cls == cls
		return torch.mean(loss[mask])

	def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		y_cls = torch.argmax(y, dim=1)
		loss = self.loss_fn(y_hat, y)

		out = torch.zeros((self.n,))
		for cls in range(self.n):
			out[cls] = self.__get_cls_loss(cls, loss, y_cls)

		out[torch.isnan(out)] = self.nan_to

		return out


