import os.path
import typing
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.utils.logger import Logger


class MLPLEvaluator:

	def __init__(
			self,
			loss: nn.Module,
			dataloader: DataLoader,
			progress_interval: int = 100,
			losses_export_path: str = None
	):
		self.__loss = loss
		self.__dataloader = dataloader
		self.__progress_interval = progress_interval
		self.__losses_export_path = losses_export_path

	def __generate_save_path(self):
		if os.path.isdir(self.__losses_export_path):
			return os.path.join(self.__losses_export_path, f"{datetime.now().timestamp()}.npy")
		return self.__losses_export_path

	def __evaluate_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
		y = y[:, 1:]
		y_hat = model(X)[:, :-1]
		return self.__loss(y_hat, y)

	def __evaluate_batch(self, models: typing.List[nn.Module], X: torch.Tensor, y: torch.Tensor) -> float:

		return torch.stack([
			self.__evaluate_model(model, X,  y)
			for model in models
		], dim=1)

	def __collapse_losses(self, losses: torch.Tensor) -> torch.Tensor:
		return torch.min(
			losses,
			dim=1
		).values

	def __get_data_size(self):
		if hasattr(self.__dataloader, "dataset"):
			return len(self.__dataloader.dataset)
		return len(self.__dataloader) * self.__dataloader.batch_size

	def __export_losses(self, losses: torch.Tensor):
		path = self.__generate_save_path()
		Logger.info(f"Exporting losses to {path}...")
		np.save(path, losses.cpu().numpy())

	def evaluate(self, models: typing.List[nn.Module]):
		for model in models:
			model.eval()
		with torch.no_grad():

			losses = torch.zeros((self.__get_data_size(), len(models)))

			for i, (X, y) in enumerate(self.__dataloader):

				losses[i * self.__dataloader.batch_size:(i + 1) * self.__dataloader.batch_size] = self.__evaluate_batch(models, X, y)

				if i % self.__progress_interval == 0:
					Logger.info(f"Evaluating: {i} / {len(self.__dataloader)}...", end="\r")

		if self.__losses_export_path is not None:
			self.__export_losses(losses)

		losses = self.__collapse_losses(losses)
		return torch.mean(losses)
