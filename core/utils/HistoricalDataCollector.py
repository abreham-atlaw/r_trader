from typing import *
from abc import ABC, abstractmethod

import pandas as pd

from polygon import RESTClient
from polygon.rest.models.definitions import CurrenciesAggregatesApiResponse

import datetime
import time
import os
import gc
from dataclasses import dataclass
from requests import HTTPError

from lib.network.oanda import Trader
from lib.network.oanda.data.models import CandleStick


@dataclass
class DataPoint:
	v: int
	o: float
	h: float
	l: float
	c: float
	time: datetime


class DataFetcher(ABC):

	@abstractmethod
	def _fetch_max(
			self,
			from_: datetime.datetime,
			instrument: Tuple[str, str],
	) -> List[DataPoint]:
		pass

	def fetch(
			self,
			from_: datetime.datetime,
			instrument: Tuple[str, str],
			to: datetime.datetime = None
	) -> pd.DataFrame:
		df = pd.DataFrame(columns=list(DataPoint(*tuple([None] * 6)).__dict__.keys()))
		start_time = from_
		previous_start_time = None

		while to > start_time and previous_start_time != start_time:
			try:
				datapoints = self._fetch_max(start_time, instrument)
			except DatasetNotFoundException:
				break
			df = df.append(pd.DataFrame([
				dp.__dict__
				for dp in datapoints
			]))
			previous_start_time = start_time
			start_time = datapoints[-1].time.replace(tzinfo=None)
			del datapoints
			gc.collect()

		return df


class PolygonDataFetcher(DataFetcher):

	def __init__(self, api_key: str):
		self.__client = RESTClient(api_key)

	@staticmethod
	def __response_to_datapoints(response: CurrenciesAggregatesApiResponse) -> List[DataPoint]:
		return [
			DataPoint(
				v=result["v"],
				o=result["o"],
				l=result["l"],
				h=result["h"],
				c=result["c"],
				time=result["t"]
			)
			for result in response.results
		]

	def _fetch_max(self, from_: datetime.datetime, instrument: Tuple[str, str]) -> List[DataPoint]:
		try:
			response: CurrenciesAggregatesApiResponse = self.__client.forex_currencies_aggregates(
				ticker=f"C:{instrument[0]}{instrument[1]}",
				multiplier=1,
				timespan="minute",
				from_=from_,
				to=datetime.datetime.now()
			)
			if response.resultsCount == 0 or response.results[-1]["t"] == from_:
				raise DatasetNotFoundException()
			return self.__response_to_datapoints(response)
		except HTTPError:
			print("[-]HttpError: Sleeping...")
			time.sleep(60)
			return self._fetch_max(from_, instrument)


class OandaDataFetcher(DataFetcher):

	def __init__(self, trader: Trader):
		self.__trader = trader

	@staticmethod
	def __candlesticks_to_datapoints(candlesticks: List[CandleStick]) -> List[DataPoint]:
		return [
			DataPoint(
				v=cs.volume,
				o=cs.mid["o"],
				l=cs.mid["l"],
				h=cs.mid["h"],
				c=cs.mid["c"],
				time=cs.time
			)
			for cs in candlesticks
		]

	def __fetch_candlestick(self, instrument, length, from_) -> List[CandleStick]:
		try:
			candlesticks = self.__trader.get_candlestick(instrument, count=length, from_=from_, granularity="M1")
			if candlesticks is None:
				raise ValueError("Candlestick is None")
		except (ValueError, HTTPError) as ex:
			print(f"Error {ex}. \nRetrying...", )
			time.sleep(5)
			return self.__fetch_candlestick(instrument, length, from_)

	def _fetch_max(self, from_: datetime.datetime, instrument: Tuple[str, str]) -> List[DataPoint]:
		candlesticks = self.__fetch_candlestick(instrument, 500, from_)
		return self.__candlesticks_to_datapoints(candlesticks)


class DataCollector:

	def __init__(self, fetcher: DataFetcher, output_dir, resume=True, merge=True):
		self.__fetcher = fetcher
		self._output_dir = output_dir
		self._resume = resume
		self.__merge = merge

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

	def collect_data(
			self,
			instruments: List[Tuple[str, str]],
			from_: datetime.datetime =datetime.datetime(year=2000, month=1, day=1),
			to: datetime.datetime = None
	):
		print("[+]Collecting Data...")

		if to is None:
			to = datetime.datetime.now()

		for base_currency, quote_currency in instruments:
			if self._resume and os.path.exists(self.__generate_save_path(base_currency, quote_currency)):
				print(f"[+]Skipping {base_currency}/{quote_currency}. Already Downloaded.")
				continue
			new_df = self.__fetcher.fetch(from_, (base_currency, quote_currency), to)
			new_df["base_currency"] = base_currency
			new_df["quote_currency"] = quote_currency
			self._save(base_currency, quote_currency, new_df)

		if self.__merge:
			merged_df = self.__merge_data(instruments)
			self._save("All", "All", merged_df)

	def __generate_save_path(self, base_currency, quote_currency):
		return os.path.join(self._output_dir, f"{base_currency}-{quote_currency}.csv")

	def _save(self, base_currency, quote_currency, df):
		output_path = self.__generate_save_path(base_currency, quote_currency)
		print(f"[+]Saving {output_path}")
		df.to_csv(output_path)


class DatasetNotFoundException(Exception):
	pass


if __name__ == "__main__":
	from core import Config

	trader = Trader(Config.OANDA_TOKEN, Config.OANDA_TEST_ACCOUNT_ID, timezone=Config.TIMEZONE)
	instruments = trader.get_instruments()

	fetcher = OandaDataFetcher(trader)
	data_collector = DataCollector(fetcher, "../../Temp/Data", resume=True)
	data_collector.collect_data(instruments)
