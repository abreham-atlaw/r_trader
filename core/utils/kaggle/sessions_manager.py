import typing

import os
import json
from datetime import datetime

from .data.repositories import SessionsRepository
from .data.models import Account, Session


class SessionsManager:

	def __init__(
			self,
			sessions_repository: SessionsRepository,
	):
		self.__session_repository = sessions_repository

	def __register_session(self, kernel: str, account: Account, gpu: bool):
		self.__session_repository.add_session(Session(
			account=account,
			kernel=kernel,
			gpu=gpu,
			active=True,
			start_datetime=datetime.now()
		))

	@staticmethod
	def __create_api(account: Account):
		os.environ["KAGGLE_USERNAME"] = account.username
		os.environ["KAGGLE_KEY"] = account.key
		from kaggle.api.kaggle_api_extended import KaggleApi
		api = KaggleApi()
		api.authenticate()
		return api

	@staticmethod
	def __update_meta(meta_data, path):
		if len(meta_data) == 0:
			return

		meta_path = os.path.join(path, "kernel-metadata.json")
		with open(meta_path, "r") as file:
			meta = json.load(file)

		meta.update(meta_data)

		with open(meta_path, "w") as file:
			json.dump(meta, file)

	@staticmethod
	def __pull_notebook(api, kernel: str) -> str:
		pull_path = f".notebook-{kernel}".replace("/", "-")
		os.mkdir(pull_path)
		api.kernels_pull(kernel, pull_path, metadata=True)
		return pull_path

	@staticmethod
	def __push_notebook(api, path):
		api.kernels_push(path)

	@staticmethod
	def __clean(path):
		os.system(f"rm -fr \"{path}\"")

	def __run_notebook(
			self,
			kernel: str,
			account: Account,
			meta_data: typing.Dict[str, typing.Any],
			gpu: bool
	):
		meta_data["enable_gpu"] = gpu
		api = self.__create_api(account)
		path = self.__pull_notebook(api, kernel)
		try:
			self.__update_meta(meta_data, path)
			self.__push_notebook(api, path)
		finally:
			self.__clean(path)

	def start_session(
			self,
			kernel: str,
			account: Account,
			meta_data: typing.Dict[str, typing.Any],
			gpu: bool = True
	):
		print(f"Running {kernel} on {account.username}(gpu={gpu})")
		self.__run_notebook(kernel, account, meta_data, gpu)
		self.__register_session(kernel, account, gpu)

	def finish_session(self, kernel: str):
		self.__session_repository.finish_session(self.__session_repository.filter(kernel=kernel, active=True)[0])