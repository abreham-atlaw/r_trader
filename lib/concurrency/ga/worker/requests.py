from typing import *

import attr
import json

from lib.network.rest_interface import Request, Serializer


@attr.define
class EvaluateResponse:
	key: str = attr.ib()
	species: Dict = attr.ib()


class EvaluateRequest(Request):

	def __init__(self):
		super().__init__(
			url="evaluate",
			method=Request.Method.GET,
			output_class=EvaluateResponse
		)


class ResponseRequest(Request):

	def __init__(self, key: str, value: float):
		super().__init__(
			url="result",
			post_data=json.dumps({
				"key": key,
				"value": value
			}),
			method=Request.Method.POST
		)
