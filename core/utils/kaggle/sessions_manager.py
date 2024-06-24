import typing

import os
import json
from datetime import datetime

from .data.repositories import SessionsRepository, AccountsRepository
from .data.models import Account, Session


class SessionsManager:

	def __init__(
			self,
			sessions_repository: SessionsRepository,
			account_repository: AccountsRepository
	):
		self.__session_repository = sessions_repository
		self.__account_repository = account_repository

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
		SessionsManager.__clean(pull_path)
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
		print(f"Starting {kernel} on {account.username}(gpu={gpu})...")
		meta_data["enable_gpu"] = gpu
		meta_data["enable_internet"] = True
		api = self.__create_api(account)
		path = self.__pull_notebook(api, kernel)
		try:
			self.__update_meta(meta_data, path)
			self.__push_notebook(api, path)
		finally:
			self.__clean(path)

	def __prepare_for_run(self, kernel: str):
		print(f"Preparing to run {kernel}...")
		self.finish_session(kernel, multiple=True)

	def start_session(
			self,
			kernel: str,
			account: Account,
			meta_data: typing.Dict[str, typing.Any],
			gpu: bool = True,
			close_others: bool = True
	):
		self.sync_notebooks()
		print(f"Running {kernel} on {account.username}(gpu={gpu})...")
		self.__prepare_for_run(kernel)
		self.__run_notebook(kernel, account, meta_data, gpu)
		self.__register_session(kernel, account, gpu)

	def finish_session(self, kernel: str, multiple=False):
		print(f"Finishing {kernel} with multiple={multiple}")
		active_sessions = self.__session_repository.filter(kernel=kernel, active=True)
		if not multiple:
			active_sessions = active_sessions[:1]
		for session in active_sessions:
			self.__session_repository.finish_session(session)

	def is_notebook_running(self, notebook_id):
		print(f"Checking {notebook_id}")
		username, slug = notebook_id.split("/")
		api = self.__create_api(self.__account_repository.get_accounts()[0])
		status = api.kernel_status(username, slug)
		return status.get("status") == "running"

	def sync_notebooks(self):
		sessions = self.__session_repository.filter(active=True)
		print(f"Syncing {len(sessions)} sessions")
		for session in sessions:
			print(f"Syncing {session.kernel}...")
			if session.account is None:
				print(f"Invalid session skipping...")
				continue
			if not self.is_notebook_running(session.kernel):
				print(f"Finishing session")
				self.finish_session(session.kernel, multiple=True)
