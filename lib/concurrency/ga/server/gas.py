from typing import *

from flask import Flask, Blueprint, request

import os
import json

from .repository import DataRepository, PlainDataRepository


class GAServer:

	def __init__(self, host: str, port: int):
		self.__app = Flask(__name__)
		self.__host, self.__port = host, port
		self.__repository = self._init_repository()

	def _init_repository(self) -> DataRepository:
		return PlainDataRepository()

	def _map_urls(self) -> List[Tuple[str, List[Tuple[str, Callable, List[str]]]]]:

		return [
			("/queen", [
				("evaluate", self.__handle_new_evaluate_request, ["POST"]),
				("result", self.__handle_result_request, ["GET"]),
				("reset", self.__handle_reset, ["GET"])
			]),
			("/worker", [
				("evaluate", self.__handle_evaluate_request, ["GET"]),
				("result", self.__handle_evaluate_response, ["POST"])
			]),
		]

	def __map_urls(self):
		for bp_url, mapping in self._map_urls():
			for url, handler, methods in mapping:
				self.__app.add_url_rule(os.path.join(bp_url, url), view_func=handler, methods=methods)

	def __handle_new_evaluate_request(self):
		self.__repository.add_to_request_queue(request.json["key"], request.json["species"])
		return "", 200

	def __handle_result_request(self):
		result = self.__repository.get_response(request.args.get("key"))
		if result is not None:
			return json.dumps(result)
		return "", 404

	def __handle_reset(self):
		self.__repository.reset()
		return "", 200

	def __handle_evaluate_request(self):
		result = self.__repository.get_request()
		if result is None:
			return "", 404
		return json.dumps({
			"key": result[0],
			"species": result[1]
		})

	def __handle_evaluate_response(self):
		self.__repository.set_response(
			request.json["key"],
			request.json["value"]
		)
		return "", 200

	def get_app(self):
		return self.__app

	def setup(self):
		self.__map_urls()

	def start(self):
		self.setup()
		self.__app.run(
			host=self.__host,
			port=self.__port
		)
