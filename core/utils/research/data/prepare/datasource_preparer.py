import os.path
import typing
from datetime import datetime, timedelta

import pandas as pd

from lib.utils.logger import Logger


class DatasourcePreparer:

	def __init__(
			self,
			export_path: str,

	):
		self.__export_path = export_path

	def __load_df(self, path):
		df = pd.read_csv(path)
		df["time"] = pd.to_datetime(df["time"])
		df = df.drop_duplicates(subset="time")
		df = df.sort_values(by="time")
		return df

	def __correct_time(self, df):
		final_time = df["time"].max()
		time_col = list(reversed([final_time - timedelta(minutes=i) for i in range(df.shape[0])]))
		df["time"] = time_col
		return df

	def __generate_export_path(self, path):
		if os.path.isdir(self.__export_path):
			return os.path.join(self.__export_path, os.path.basename(path))

		return self.__export_path

	def __export_df(self, df, import_path):
		path = self.__generate_export_path(import_path)
		Logger.info(f"[+]Exporting {path} ...")
		df.to_csv(path, index=False)

	def prepare(
			self,
			path: str,
			time_range: typing.Tuple[datetime, datetime]
	):

		df = self.__load_df(path)
		df = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1])]
		df = self.__correct_time(df)
		self.__export_df(df, path)
