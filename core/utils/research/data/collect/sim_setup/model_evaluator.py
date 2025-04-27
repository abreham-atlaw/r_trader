import os
import typing

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.losses import ProximalMaskedLoss
from core.utils.research.training.trainer import Trainer


class ModelEvaluator:

	def __init__(
			self,
			data_path: typing.Union[str, typing.List[str]],
			loss_fn: nn.Module,
			batch_size: int = 32,
	):
		if isinstance(data_path, str):
			data_path = [data_path]
		self.__loss_fn = loss_fn
		self.__data_path, self.__batch_size = data_path, batch_size

	@staticmethod
	def __init_dataloader(paths: typing.List[str], batch_size: int):
		dataset = dataset = BaseDataset(
			root_dirs=paths,
			out_dtypes=np.float32,
			check_file_sizes=True
		)
		dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
		return dataloader

	def __init_trainer(self, model):
		trainer = Trainer(model)
		trainer.cls_loss_function = self.__loss_fn
		trainer.reg_loss_function = nn.MSELoss()
		return trainer

	def evaluate(self, model) -> typing.Tuple[float, float, float]:
		model.eval()

		trainer = self.__init_trainer(model)

		dataloader = self.__init_dataloader(self.__data_path, self.__batch_size)

		losses = trainer.validate(dataloader)
		return losses
