import typing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MLPLEvaluator:

	def __init__(self, loss: nn.Module, dataloader: DataLoader):
		self.__loss = loss
		self.__dataloader = dataloader

	def __evaluate_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
		y_hat = model(X)
		return self.__loss(y_hat, y)

	def __evaluate_batch(self, models: typing.List[nn.Module], X: torch.Tensor, y: torch.Tensor) -> float:
		return min([
			self.__evaluate_model(model, X, y)
			for model in models
		])

	def evaluate(self, models: typing.List[nn.Module]):
		for model in models:
			model.eval()
		with torch.no_grad():
			losses = [
				self.__evaluate_batch(models, X, y)
				for X, y in self.__dataloader
			]
			return torch.mean(torch.tensor(losses))
