import os.path
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
			export_path: str
	):
		self.model = model
		self.export_path = export_path

	def __generate_path(self):
		return os.path.join(self.export_path, f"{datetime.now().timestamp()}.npy")

	def __export_output(self, output: torch.Tensor):
		np.save(
			self.__generate_path(),
			output.detach().numpy()
		)

	def export(self, dataloader: DataLoader):
		self.model.eval()
		with torch.no_grad():
			for i, (X, y) in enumerate(dataloader):
				out = self.model(X)
				self.__export_output(out)
				Logger.info(f"Exported {(i+1)*100/len(dataloader):.2f}%...")
