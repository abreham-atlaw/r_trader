from typing import *

import json

from lib.network.rest_interface.requests import Request


class EvaluateRequest(Request):

	def __init__(self, key: str, species: dict):
		super().__init__(
			url="evaluate",
			post_data=json.dumps({
				"key": key,
				"species": species
			}),
			method=Request.Method.POST
		)


class GetResult(Request):

	def __init__(self, key: str):
		super().__init__(
			url="result",
			get_params={
				"key": key
			},
			method=Request.Method.GET,
			output_class=float
		)
