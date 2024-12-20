import os.path
import typing
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger


class ModelOutputExporter:

	def __init__(
			self,
			model: SpinozaModule,
			export_path: str,
			X_dir: str = "X",
			y_hat_dir: str = "y_hat",
			y_dir: str = "y"
	):
		self.model = model
		self.export_path = export_path
		self.__X_dir = X_dir
		self.__y_hat_dir = y_hat_dir
		self.__y_dir = y_dir

	def __setup_export_dirs(self):
		for path in [
			self.export_path,
			os.path.join(self.export_path, self.__X_dir),
			os.path.join(self.export_path, self.__y_dir),
			os.path.join(self.export_path, self.__y_hat_dir)
		]:
			if not os.path.exists(path):
				Logger.info(f"[+]Creating {path}...")
				os.makedirs(path)

	def __generate_path(self) -> typing.Tuple[str, str, str]:
		filename = f"{datetime.now().timestamp()}.npy"
		return tuple([
			os.path.join(self.export_path, dir_name, filename)
			for dir_name in [self.__X_dir, self.__y_dir, self.__y_hat_dir]
		])

	@staticmethod
	def __export_array(array: torch.Tensor, path: str):
		np.save(path, array.detach().numpy())

	def __export_output(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
		for array, path in zip([X, y, y_hat], self.__generate_path()):
			self.__export_array(array, path)

	def export(self, dataloader: DataLoader):
		self.__setup_export_dirs()
		self.model.eval()
		with torch.no_grad():
			for i, (X, y) in enumerate(dataloader):
				y_hat = self.model(X)
				self.__export_output(X, y, y_hat)
				Logger.info(f"Exported {(i+1)*100/len(dataloader):.2f}%...", end="\r")
