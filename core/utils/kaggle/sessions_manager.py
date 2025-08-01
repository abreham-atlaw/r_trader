import time
import typing

import os
import json
from datetime import datetime

from lib.utils.logger import Logger
from .data.repositories import SessionsRepository, AccountsRepository
from .data.models import Account, Session, Resources


class SessionsManager:

	def __init__(
			self,
			sessions_repository: SessionsRepository,
			account_repository: AccountsRepository,
			dataset_key_alias: bool = True,
			retry_patience: int = 10
	):
		self.__session_repository = sessions_repository
		self.__account_repository = account_repository
		self.__dataset_key_alias = dataset_key_alias
		self.__retry_patience = retry_patience

	def __register_session(self, kernel: str, account: Account, device: int):
		self.__session_repository.add_session(Session(
			account=account,
			kernel=kernel,
			device=device,
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
			device: int
	):
		Logger.info(f"Starting {kernel} on {account.username}(device={device})...")
		meta_data = meta_data.copy()

		meta_data["enable_gpu"] = device == Resources.Devices.GPU
		meta_data["enable_tpu"] = device == Resources.Devices.TPU

		meta_data["enable_internet"] = True
		if self.__dataset_key_alias and meta_data.get("dataset_sources") is not None:
			meta_data["kernel_sources"] = meta_data.pop("dataset_sources")
			meta_data["dataset_sources"] = []

		api = self.__create_api(account)
		path = self.__pull_notebook(api, kernel)
		try:
			self.__update_meta(meta_data, path)
			self.__push_notebook(api, path)
		finally:
			self.__clean(path)

	def __prepare_for_run(self, kernel: str):
		Logger.info(f"Preparing to run {kernel}...")
		self.finish_session(kernel, multiple=True)

	def start_session(
			self,
			kernel: str,
			account: Account,
			meta_data: typing.Dict[str, typing.Any],
			device: int,
			close_others: bool = True,
			sync_notebooks: bool = True
	):
		if sync_notebooks:
			self.sync_notebooks()
		Logger.info(f"Running {kernel} on {account.username}(device={device})...")
		self.__prepare_for_run(kernel)
		self.__run_notebook(kernel, account, meta_data, device)
		self.__register_session(kernel, account, device)

	def finish_session(self, kernel: str, multiple=False):
		Logger.info(f"Finishing {kernel} with multiple={multiple}")
		active_sessions = self.__session_repository.filter(kernel=kernel, active=True)
		if not multiple:
			active_sessions = active_sessions[:1]
		for session in active_sessions:
			self.__session_repository.finish_session(session)

	def __get_notebook_status(self, api, username, slug, tries=0):
		from kaggle.rest import ApiException
		try:
			return api.kernel_status(username, slug)
		except ApiException as ex:
			if ex.status in [429, 404] and tries < self.__retry_patience:
				Logger.warning(f"Rate limited[{ex.status}]. Waiting 30 seconds...")
				time.sleep(30)
				return self.__get_notebook_status(api, username, slug, tries=tries+1)
			else:
				raise ex

	def is_notebook_running(self, notebook_id):
		Logger.info(f"Checking {notebook_id}")
		username, slug = notebook_id.split("/")
		api = self.__create_api(self.__account_repository.get_accounts()[0])
		status = self.__get_notebook_status(api, username, slug)
		Logger.info(f"{notebook_id} status: {status.get('status')}")
		return status.get("status") == "running"

	def sync_notebooks(self):
		sessions = self.__session_repository.filter(active=True)
		Logger.info(f"Syncing {len(sessions)} sessions")
		for session in sessions:
			Logger.info(f"Syncing {session.kernel}...")
			if session.account is None:
				Logger.info(f"Invalid session skipping...")
				continue
			try:
				if not self.is_notebook_running(session.kernel):
					Logger.info(f"Finishing session")
					self.finish_session(session.kernel, multiple=True)
			except Exception as ex:
				Logger.error(f"Failed to finish session {session.kernel}")
				Logger.error(ex)
