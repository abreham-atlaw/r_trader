from datetime import datetime
from typing import Dict

from lib.network.oanda.data.models import AccountSummary
from lib.network.rest_interface import Request


class CreateAccountRequest(Request):

	def __init__(
			self,
			start_time: datetime,
			delta_multiplier: float,
			margin_rate: float,
			alias: str,
			balance: float
	):
		super().__init__(
			"accounts/create/",
			method=Request.Method.POST,
			post_data={
				"start_time": start_time.strftime("%Y-%m-%d %H:%M:%S+00:00"),
				"delta_multiplier": delta_multiplier,
				"margin_rate": margin_rate,
				"alias": alias,
				"balance": balance
			},
			output_class=AccountSummary
		)

	def _filter_response(self, response):
		return response["account"]
