import os
import typing
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from core import Config
from core.di import ServiceProvider
from core.utils.research.training.data.ims import IMS, IMSs
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.file_storage import FileStorage


class IMSRepository:

	def __init__(
			self,
			model_name: str,
			label: str,
			fs: FileStorage = None,
			stats: typing.List[IMS] = None,
			sync_size: int = 1024,
			tmp_path: str = "./",
			drop_idx: bool = True
	):

		self.__dump_path = None
		self.__fs = fs if fs is not None else ServiceProvider.provide_file_storage()
		self.__stats = stats if stats is not None else IMSs.all
		self.__model_name = model_name
		self.__label = label
		self.__dump_size = sync_size
		self.__df = self.__init_df()
		self.__tmp_path = tmp_path
		self.__drop_idx = drop_idx

	def __init_df(self):
		return pd.DataFrame(columns=[stat.name for stat in self.__stats])

	def __reset_df(self):
		self.__df = self.__init_df()

	@property
	def __id(self) -> str:
		return f"{self.__model_name}.{self.__label}"

	def __generate_save_path(self) -> str:

		# bound_label = '-'.join([
		# 	','.join([
		# 		str(self.__df.iloc[i][c])
		# 		for c in ['epoch', 'batch']
		# 	])
		# 	for i in [0, -1]
		# ])

		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

		return os.path.join(
			self.__tmp_path,
			f"{self.__id}.{timestamp}.csv"
		)

	def __generate_download_path(self, filepath: str) -> str:
		return os.path.join(
			self.__tmp_path,
			os.path.basename(filepath)
		)

	def sync(self):
		save_path = self.__generate_save_path()
		self.__df.to_csv(save_path, index=False)
		self.__fs.upload_file(save_path)
		self.__reset_df()

	def __generate_stats(self, value: np.ndarray) -> typing.Dict:
		return {
			stat.name: stat.compute(value)
			for stat in self.__stats
		}

	@CacheDecorators.cached_method()
	def __load_df(self, filename: str) -> pd.DataFrame:
		download_path = self.__generate_download_path(filename)
		if not os.path.exists(download_path):
			self.__fs.download(filename, download_path=download_path)
		df = pd.read_csv(download_path, index_col=None)
		if self.__drop_idx:
			df = df.reset_index(drop=True)
			# batch_size = max(df["batch"])
			# df["index"] = df.apply(lambda row: int(row["epoch"] * batch_size + row["batch"]), axis=1)
			# df = df.set_index("index")
		return df

	def __load_dfs(self) -> pd.DataFrame:
		files = sorted(list(filter(
			lambda filepath: os.path.basename(filepath).startswith(self.__id),
			self.__fs.listdir("")
		)))
		dfs = [
			self.__load_df(filename)
			for filename in files
		]
		if len(dfs) == 0:
			return self.__init_df()

		df = pd.concat(dfs)
		if self.__drop_idx:
			df = df.reset_index(drop=True)
		return df

	def store(
			self,
			value: typing.Union[np.ndarray, torch.Tensor],
			epoch: int,
			batch: int
	):

		if isinstance(value, torch.Tensor):
			value = value.detach().cpu().numpy()

		values = self.__generate_stats(value)
		values["epoch"], values["batch"] = epoch, batch
		if self.__df.shape[0] == 0:
			self.__df = pd.DataFrame(values, index=[0])
		else:
			self.__df = pd.concat([self.__df, pd.DataFrame(values, index=[0])])
		if self.__df.shape[0] >= self.__dump_size:
			self.sync()

	def retrieve(self) -> pd.DataFrame:
		dfs = self.__load_dfs()
		return dfs

	@staticmethod
	def retrieve_labels(
			model_name: str,
			fs: FileStorage = None
	) -> typing.List[str]:
		if fs is None:
			fs = ServiceProvider.provide_file_storage(Config.IMS_REMOTE_PATH)
		files = sorted(list(filter(
			lambda filepath: os.path.basename(filepath).startswith(model_name),
			fs.listdir("")
		)))
		return list(map(
			lambda filename: os.path.basename(filename).split(".")[1],
			files
		))

