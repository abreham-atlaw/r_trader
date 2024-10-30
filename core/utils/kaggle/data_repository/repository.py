import hashlib
import os
import typing

from requests.exceptions import ChunkedEncodingError

from core.di import ServiceProvider
from core.utils.kaggle.data.models import Account
from core.utils.kaggle.data.repositories import MongoAccountsRepository
from lib.utils.decorators import retry
from lib.utils.logger import Logger
from .exceptions import IntegrityFailException


class KaggleDataRepository:

	def __init__(
			self,
			account: Account = None,
			output_path: str = "./",
			zip_filename: str = "_output_.zip",
	):
		self.__account_singleton = account
		self.__api_singleton = None
		self.__output_path = output_path
		self.__zip_filename = zip_filename

	@property
	def __account(self) -> Account:
		if self.__account_singleton is not None:
			return self.__account_singleton
		self.__account_singleton = MongoAccountsRepository(
			ServiceProvider.provide_mongo_client()
		).get_accounts()[0]
		Logger.info(f"Using Account: {self.__account_singleton.username}")
		return self.__account_singleton

	@property
	def __api(self):
		if self.__api_singleton is not None:
			return self.__api_singleton
		os.environ['KAGGLE_USERNAME'] = self.__account.username  # Replace with your username
		os.environ['KAGGLE_KEY'] = self.__account.key
		from kaggle.api.kaggle_api_extended import KaggleApi
		api = KaggleApi()
		api.authenticate()
		return api

	@staticmethod
	def __unzip(zip_path: str, output_path: str):
		os.system(f"unzip -d \"{output_path}\" \"{zip_path}\"")

	def __generate_download_path(self, kernel: str) -> str:
		return os.path.join(self.__output_path, kernel.replace("/", "-"))

	@staticmethod
	def __clean(path: str):
		Logger.info(f"Cleaning {path}")
		os.system(f"rm -fr \"{path}\"")

	def __download_kernel(self, kernel, download_path):
		self.__api.kernels_output(kernel, download_path)

	def __download_dataset(self, dataset, download_path):
		self.__api.dataset_download_files(dataset=dataset, path=download_path)

	@retry(exception_cls=(ChunkedEncodingError, IntegrityFailException), patience=10, sleep_timer=5)
	def download(self, slug: str, kernel=True, checksum: str = None) -> str:
		Logger.info(f"Downloading {slug}")
		download_path = self.__generate_download_path(slug)
		Logger.info(f"Downloading to {download_path}")
		self.__clean(download_path)
		os.mkdir(download_path)
		if kernel:
			self.__download_kernel(slug, download_path)
		else:
			self.__download_dataset(slug, download_path)
		zip_names = [self.__zip_filename, f"{slug.split('/')[1]}.zip"]

		for zip_name in zip_names:
			if os.path.exists(os.path.join(download_path, zip_name)):
				Logger.info(f"Unzipping Data...")
				os.system(f"unzip -d \"{download_path}\" \"{download_path}/{zip_name}\"")

		if checksum is not None:
			if not self.check_integrity(download_path, checksum):
				raise IntegrityFailException()

		return download_path

	def download_multiple(self, kernels: typing.List[str], *args, **kwargs) -> typing.List[str]:
		return [
			self.download(slug=kernel, *args, **kwargs)
			for kernel in kernels
		]

	@staticmethod
	def generate_checksum(path: str) -> str:
		Logger.info(f"Generating checksum for '{path}'")
		files_str = str(list(os.walk(path)))
		return hashlib.sha256(files_str.encode('utf-8')).hexdigest()

	def check_integrity(self, path, checksum: str) -> bool:
		return self.generate_checksum(path) == checksum
