import os
import typing

from core.di import ServiceProvider
from core.utils.kaggle.data.models import Account
from core.utils.kaggle.data.repositories import MongoAccountsRepository


class KaggleDataRepository:


	def __init__(
			self,
			account: Account = None,
			output_path: str = "./",
			zip_filename: str = "_output_.zip"
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
		print(f"Using Account: {self.__account_singleton.username}")
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

	def __generate_download_path(self, kernel: str) -> str:
		return os.path.join(self.__output_path, kernel.replace("/", "-"))

	@staticmethod
	def __clean(path: str):
		os.system(f"rm -fr \"{path}\"")

	def __download_kernel(self, kernel, download_path):
		self.__api.kernels_output(kernel, download_path)

	def __download_dataset(self, dataset, download_path):
		raise Exception("Unimplemented...")

	def download(self, slug: str, kernel=True) -> str:
		print(f"Downloading {slug}")
		download_path = self.__generate_download_path(slug)
		print(f"Downloading to {download_path}")
		self.__clean(download_path)
		os.mkdir(download_path)
		if kernel:
			self.__download_kernel(slug, download_path)
		else:
			self.__download_dataset(slug, download_path)
		if os.path.exists(os.path.join(download_path, self.__zip_filename)):
			print(f"Unzipping Data...")
			os.system(f"unzip -d \"{download_path}\" \"{download_path}/{self.__zip_filename}\"")
		return download_path

	def download_multiple(self, kernels: typing.List[str]) -> typing.List[str]:
		return [
			self.download(slug=kernel)
			for kernel in kernels
		]
