from typing import *

import requests
from requests.exceptions import HTTPError
import json

from . import Request
from lib.network import network_call
from .exceptions import InvalidNetworkMethod


class NetworkApiClient:

	def __init__(self, url: str):
		self.__url = url
		if url.endswith("/"):
			self.__url = url[:-1]

	def _get_complete_url(self, url):
		return f"{self.__url}/{url}"

	@network_call
	def _get(self, request: Request, headers=None):
		return requests.get(self._get_complete_url(request.get_url()), params=request.get_get_params(), headers=headers)

	@network_call
	def _post(self, request: Request, headers=None):
		return requests.post(self._get_complete_url(request.get_url()), data=request.get_post_data(), headers=headers)

	@network_call
	def _put(self, request: Request, headers=None):
		return requests.put(self._get_complete_url(request.get_url()), data=request.get_post_data(), headers=headers)

	def execute(self, request: Request, headers: Dict=None):
		response = None
		headers.update(request.get_headers())
		if request.get_method() == Request.Method.GET:
			response = self._get(request, headers=headers)
		elif request.get_method() == Request.Method.POST:
			response = self._post(request, headers=headers)
		elif request.get_method() == Request.Method.PUT:
			response = self._put(request, headers=headers)
		else:
			raise InvalidNetworkMethod()

		if 400 <= response.status_code < 600:
			raise HTTPError(f"Status Code: {response.status_code} Message: {response.json()}")
		return request.deserialize_object(response.json())