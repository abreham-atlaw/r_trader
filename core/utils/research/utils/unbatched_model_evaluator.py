import typing

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from lib.utils.logger import Logger
from .model_evaluator import ModelEvaluator


class UnbatchedModelEvaluator(ModelEvaluator):

	def __init__(self, *args, output_slice: typing.Tuple[int, int] = None, **kwargs):
		super().__init__(*args, **kwargs)
		if output_slice is None:
			output_slice = (0, -1)
		self.__output_slice = output_slice

	def _evaluate(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module):

		losses = torch.Tensor([])

		for i, (X, y) in enumerate(dataloader):
			with torch.no_grad():
				y_hat = model(X)

			y, y_hat = [arr[:, self.__output_slice[0]:self.__output_slice[1]] for arr in [y, y_hat]]

			loss = loss_fn(y_hat, y)

			losses = torch.cat((losses, loss), dim=0)

			if i % 10 == 0:
				Logger.info(f"Evaluating batch {i + 1}/{len(dataloader)}", end="\r")

		return losses
