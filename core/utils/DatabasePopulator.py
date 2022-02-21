from typing import *

import pandas as pd

import psycopg2 as pg
import os
from datetime import datetime

from core import Config

FILES_DIR = os.path.abspath("../../Data/Minutes")

# TEMP: REMOVE IT
VALID_CURRENCIES = [
	"AUD",
	"GBP",
	"CAD",
	"USD",
]


class Populator:

	def __init__(self, file_pathes: List[str], db_config: Dict, table_name: str):
		self.__file_pathes = file_pathes
		self.__connection = pg.connect(**db_config)
		self.__table_name = table_name

	def __load_file(self, file_path: str) -> pd.DataFrame:
		return pd.read_csv(file_path)

	def __populate_entry(self, row, cursor=None, commit=False):
		if cursor is None:
			cursor = self.__connection.cursor()
		cursor.execute(
			f"INSERT INTO {self.__table_name}(base_currency, quote_currency, close, high, low, open, datetime, volume) "
			f"values(%s, %s, %s, %s, %s, %s, %s, %s);",
			(row["base_currency"], row["quote_currency"], row["c"], row["h"], row["l"], row["o"], datetime.fromtimestamp(row["t"]/1000), row["v"])
		)
		if commit:
			self.__connection.commit()

	def __is_already_populated(self, base_currency, quote_currency, cursor=None) -> bool:
		if cursor is None:
			cursor = self.__connection.cursor()

		cursor.execute(f"SELECT * FROM {self.__table_name} WHERE base_currency = %s AND quote_currency = %s;", (base_currency, quote_currency))
		response = cursor.fetchall()
		return len(response) > 0

	def __populate_database(self, data: pd.DataFrame):
		cursor = self.__connection.cursor()
		for i, row in data.iterrows():
			self.__populate_entry(row, cursor, False)
			print(f"\rCompleted {(i+1)*100/len(data):.2f}%..", end="")
		self.__connection.commit()

	def start(self):
		for file_path in self.__file_pathes:
			print(f"Populating {file_path}")
			base_currency, quote_currency = file_path.split("/")[-1].split(".")[0].split("-")
			if self.__is_already_populated(base_currency, quote_currency):
				print(f"{file_path} already populated. Skipping...")
				continue
			self.__populate_database(
				self.__load_file(file_path)
			)


if __name__ == "__main__":
	#files_path = [os.path.join(FILES_DIR, file) for file in os.listdir(FILES_DIR)]
	files_path = [
		os.path.join(
			FILES_DIR,
			f"{base_currency}-{quote_currency}.csv"
		)
		for i, base_currency in enumerate(VALID_CURRENCIES)
		for quote_currency in VALID_CURRENCIES[i+1:]
		if base_currency != quote_currency
	]

	populator = Populator(files_path, Config.DEFAULT_PG_CONFIG, Config.HISTORICAL_TABLE_NAME)
	populator.start()


