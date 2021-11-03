from typing import *

import pandas as pd

from polygon import RESTClient
from polygon.rest.models.definitions import CurrenciesAggregatesApiResponse

from requests import HTTPError

import datetime
import time
import os

API_KEY = "1ijeQ0XUYNl1YMHy6Wl_5zEBtGbkipUP"
CURRENCIES = [
	"AUD",
	"CAD",
	"CHF",
	"CZK",
	"DKK",
	"GBP",
	"HKD",
	"HUF",
	"JPY",
	"MXN",
	"NOK",
	"NZD",
	"PLN",
	"SEK",
	"SGD",
	"THB",
	"TRY",
	"USD",
	"ZAR"
]

CURRENCY_PAIRS = [
	(base_currency, quote_currency)
	for base_currency in CURRENCIES
	for quote_currency in CURRENCIES
	if base_currency != quote_currency
]

OUTPUT_LOCATION = os.path.abspath("../../Data/Minutes/")


class DataCollector:

	def __init__(self, api_key, output_dir, resume=True):
		self.__client = RESTClient(api_key)
		self._output_dir = output_dir
		self._resume = resume

	def __merge_data(self, currency_pairs) -> pd.DataFrame:

		print("[+]Merging Data...")

		df: pd.DataFrame = None

		for base_currency, quote_currency in currency_pairs:
			pair_df = pd.read_csv(self.__generate_save_path(base_currency, quote_currency), index_col=0)
			if df is None:
				df = pair_df
				continue
			df = df.append(pair_df)

		return df

	def collect_data(self, currency_pairs: List[Tuple[str, str]], time_span="minute",
					from_=datetime.datetime(year=2000, month=1, day=1),	to=datetime.date.today()) -> pd.DataFrame:

		print("[+]Collecting Data...")

		for base_currency, quote_currency in currency_pairs:
			if self._resume and os.path.exists(self.__generate_save_path(base_currency, quote_currency)):
				print(f"[+]Skipping {base_currency}/{quote_currency}. Already Downloaded.")
				continue
			new_df = self.collect_pair_data(base_currency, quote_currency, time_span, from_, to)
			new_df["base_currency"] = base_currency
			new_df["quote_currency"] = quote_currency
			self._save(base_currency, quote_currency, new_df)

		merged_df = self.__merge_data(currency_pairs)
		self._save("All", "All", merged_df)

	def collect_pair_data(self, base_currency, quote_currency, time_span, from_, to=datetime.date.today()) -> pd.DataFrame:
		print(f"[+]Collecting Data ({base_currency}/{quote_currency})")
		df = pd.DataFrame(columns=["c", "h", "l", "n", "o", "t", "v"])
		start_time = int(from_.timestamp()*1000)
		while True:
			print(f"[+]Getting from {datetime.datetime.fromtimestamp(start_time/1000)} to {to}")
			try:
				response: CurrenciesAggregatesApiResponse = self.__client.forex_currencies_aggregates(
					ticker=f"C:{base_currency}{quote_currency}",
					multiplier=1,
					timespan=time_span,
					from_=start_time,
					to=to
				)
				if response.resultsCount == 0:
					break
				potential_start_time = response.results[-1]["t"]
				if start_time == potential_start_time:
					break
				df = df.append(response.results[1:])
				start_time = potential_start_time
				if datetime.datetime.fromtimestamp(start_time/1000).date() == to:
					break
			except HTTPError:
				print("[-]HTTP Error: Sleeping...")
				time.sleep(60)
				print("[+]Waking Up...")
				continue
		print("[+]Done Collecting Data.")
		return df

	def __generate_save_path(self, base_currency, quote_currency):
		return os.path.join(self._output_dir, f"{base_currency}-{quote_currency}.csv")

	def _save(self, base_currency, quote_currency, df):
		output_path = self.__generate_save_path(base_currency, quote_currency)
		print(f"[+]Saving {output_path}")
		df.to_csv(output_path)


if __name__ == "__main__":
	data_collector = DataCollector(API_KEY, OUTPUT_LOCATION, resume=True)
	data_collector.collect_data(CURRENCY_PAIRS)
