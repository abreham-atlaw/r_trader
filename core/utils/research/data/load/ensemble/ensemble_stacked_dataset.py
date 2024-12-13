import random
import typing

import torch
from torch.utils.data import Dataset
import numpy as np

from core.utils.research.data.load import BaseDataset
from lib.utils.torch_utils.tensor_merger import TensorMerger


class EnsembleStackedDataset(Dataset):

	def __init__(
			self,
			root_dirs: typing.List[typing.List[str]],
			X_dir: str = "X",
			y_dir: str = "y",
			y_hat_dir: str = "y_hat",
			out_dtypes: typing.Type = np.float32,
	):
		super().__init__()
		self.__out_dtypes = out_dtypes
		self.__X_dir = X_dir
		self.__y_dir = y_dir
		self.__y_hat_dir = y_hat_dir
		self.__Xy_dataset, self.__models_datasets = self.__create_datasets(root_dirs)
		self.merger = TensorMerger()
		self.__indexes = list(range(len(self.__Xy_dataset)))
		self[0]

	def __create_datasets(self, root_dirs: typing.List[typing.List[str]]):
		xy_dataset = BaseDataset(
			root_dirs=root_dirs[0],
			out_dtypes=self.__out_dtypes,
			check_last_file=True
		)

		models_datasets = [
			BaseDataset(
				root_dirs=root_dir,
				X_dir=self.__y_hat_dir,
				out_dtypes=self.__out_dtypes,
				check_last_file=True
			)
			for root_dir in root_dirs
		]

		return xy_dataset, models_datasets

	def __len__(self):
		return len(self.__Xy_dataset)

	def shuffle(self):
		random.shuffle(self.__indexes)

	def __getitem__(self, idx):
		X, y = self.__Xy_dataset[self.__indexes[idx]]
		y_hat = torch.stack([dataset[idx][0] for dataset in self.__models_datasets], dim=0)

		return self.merger.merge([torch.unsqueeze(X, dim=0), y_hat]), y
