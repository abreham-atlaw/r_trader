from typing import *

import json

from . import Serializer


class Request:

	class Method:
		GET = "GET"
		POST = "POST"
		PUT = "PUT"

	def __init__(
			self,
			url: str,
			get_params: Dict = None,
			post_data: object = None,
			files: dict = None,
			method: str = Method.GET,
			output_class: type = None,
			url_params=None,
			headers: Dict = None,
			content_type: Optional[str] = "application/json"
	):
		self.__url = url
		self.__method = method
		self.__get_params = get_params
		self.__serializer = Serializer(output_class)
		self.__post_data = post_data
		self.__url_params = url_params
		self.__headers = headers
		self.__files = files

		if post_data is None:
			self.__post_data = {}
		if get_params is None:
			self.__get_params = {}
		if url_params is None:
			self.__url_params = {}
		if method is None:
			self.__method = Request.Method.GET
		if headers is None:
			self.__headers = {}

	def get_url(self) -> str:
		return self.__url.format(**self.__url_params)

	def get_files(self) -> Optional[dict]:
		return self.__files

	def get_get_params(self) -> Dict:
		return self.__get_params

	def get_post_data(self) -> Dict:
		return self.__serializer.serialize(self.__post_data)

	def get_method(self) -> str:
		return self.__method
	
	def get_headers(self) -> Dict:
		return self.__headers

	def _filter_response(self, response):
		return response

	def deserialize_object(self, response) -> object:
		return self.__serializer.deserialize(
			self._filter_response(
				response
			)
		)
