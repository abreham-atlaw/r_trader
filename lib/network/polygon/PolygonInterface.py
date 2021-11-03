from typing import *

import numpy as np
import pandas as pd
from polygon import RESTClient

import datetime
import time

from lib.network import network_call


class PolygonInterface:

	class TimeSpan:
		MINUTE = "minute"
		HOUR = "hour"
		DAY = "day"

		KWARGS_MAP = {
			"day": "days",
			"month": "months",
			"year": "years"
		}

	def __init__(self, api_key):
		self.client = RESTClient(api_key)

	@network_call
	def __get_forex_aggregates(self, ticker, multiplier, time_span, from_, to):
		return self.client.forex_currencies_aggregates(
			ticker=ticker,
			multiplier=multiplier,
			timespan=time_span,
			from_=from_,
			to=to
		)

	def get_exchanges(self, base_currency, quote_currency, time_units, time_span: str = TimeSpan.MINUTE) -> pd.DataFrame:
		if time_span not in list(PolygonInterface.TimeSpan.KWARGS_MAP.keys()):
			raise Exception(f"Invalid Timespan {time_span}")

		print("[+]Getting Aggregates...")
		response = self.__get_forex_aggregates(
			ticker=f"C:{base_currency}{quote_currency}",
			multiplier=1,
			time_span=time_span,
			from_=datetime.date.today()-datetime.timedelta(**{PolygonInterface.TimeSpan.KWARGS_MAP[time_span]: time_units}),
			to=datetime.date.today()
		)

		if len(response.results) < time_units:
			return self.get_exchanges(base_currency, quote_currency, 2*time_units - len(response.results), time_span)

		return pd.DataFrame(response.results)
