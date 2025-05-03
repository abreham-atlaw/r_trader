import numpy as np
import torch
import torch.nn as nn

from core.utils.research.data.prepare.swg import AbstractSampleWeightGenerator
from core.utils.research.losses import SpinozaLoss
from lib.utils.math import sigmoid


class IdealModelSampleWeightGenerator(AbstractSampleWeightGenerator):

	def __init__(
			self,
			*args,
			model: nn.Module,
			loss: SpinozaLoss,
			y_extra_len: int = 1,
			eps: float = 1e-6,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__model = model
		self.__loss = loss
		self.__y_extra_len = y_extra_len
		self.__eps = eps

	@staticmethod
	def __to_tensor(x) -> torch.Tensor:
		return torch.from_numpy(x.astype(np.float32))

	def __eval_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		with torch.no_grad():
			y_hat = self.__model(self.__to_tensor(X))[:, :-self.__y_extra_len]
		return self.__loss(y_hat, self.__to_tensor(y)).detach().numpy()

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		y = y[:, :-self.__y_extra_len]
		loss = self.__eval_loss(X, y)
		return sigmoid(1/(loss + self.__eps))
