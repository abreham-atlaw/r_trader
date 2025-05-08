import numpy as np
import torch

import torch.nn as nn

from core.utils.research.data.prepare.swg import AbstractSampleWeightGenerator
from core.utils.research.losses import SpinozaLoss
from lib.utils.math import sigmoid


class DisagreementSampleWeightGenerator(AbstractSampleWeightGenerator):

	def __init__(
			self,
			*args,
			loss: SpinozaLoss,
			anchor_model: nn.Module,
			weak_model: nn.Module,
			y_extra_len: int = 1,
			p: int = 1,
			m: int = 1,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__loss = loss
		self.__anchor_model = anchor_model
		self.__weak_model = weak_model
		self.__y_extra_len = y_extra_len
		self.__p = p
		self.__m = m

	@staticmethod
	def __to_tensor(x) -> torch.Tensor:
		return torch.from_numpy(x.astype(np.float32))

	def __eval_loss(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> np.ndarray:
		with torch.no_grad():
			y_hat = model(self.__to_tensor(X))[:, :-self.__y_extra_len]
		return self.__loss(y_hat, self.__to_tensor(y)).detach().numpy()

	def _generate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

		y = y[:, :-self.__y_extra_len]

		anchor_loss = self.__eval_loss(self.__anchor_model, X, y)
		weak_loss = self.__eval_loss(self.__weak_model, X, y)

		return (sigmoid(weak_loss - anchor_loss) * self.__m) ** self.__p
