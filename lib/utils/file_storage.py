from typing import *
from abc import ABC, abstractmethod

import dropbox

import os

from lib.network.rest_interface.requests import Request
from lib.network.rest_interface.NetworkApiClient import NetworkApiClient


class FileStorage(ABC):

	@abstractmethod
	def get_url(self, path) -> str:
		pass

	@abstractmethod
	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		pass

	def download(self, path, download_path: Union[str, None] = None):
		url = self.get_url(path)
		command = f"wget --no-verbose \"{url}\""
		if download_path is not None:
			command = f"{command} -O {download_path}"
		os.system(command)


class DropboxClient(FileStorage):

	def __init__(self, token, folder):
		self.__client = dropbox.Dropbox(token)
		self.__folder = folder

	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		if upload_path is None:
			upload_path = os.path.join(self.__folder, file_path.split("/")[-1])

		with open(file_path, "rb") as f:
			meta = self.__client.files_upload(f.read(), upload_path, mode=dropbox.files.WriteMode("overwrite"))
			return meta

	def get_url(self, path) -> str:
		raise Exception("UnImplemented")


class PCloudClient(FileStorage):

	class PCloudNetworkClient(NetworkApiClient):

		def __init__(self, token: str, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.__token = token

		def execute(self, request: Request, headers: Optional[Dict] = None):
			request.get_get_params()["auth"] = self.__token
			return super().execute(request, headers)

	class UploadFileRequest(Request):

		def __init__(self, filepath: str, upload_path: Optional[str] = None):
			get_params = {}
			if upload_path is not None:
				get_params["path"] = upload_path
			super().__init__(
				"uploadfile",
				method=Request.Method.POST,
				get_params=get_params,
				files={
					"file": filepath
				},
				headers={
					"Content-Type": None
				}
			)

	class CreateUrlRequest(Request):

		def __init__(self, path):
			super().__init__(
				"/getfilepublink",
				method=Request.Method.GET,
				get_params={
					"path": path
				},
				output_class=str
			)

		def _filter_response(self, response: Dict) -> str:
			return response.get("code")

	class GetUrlRequest(Request):

		def __init__(self, code: str):
			super().__init__(
				"/getpublinkdownload",
				method=Request.Method.GET,
				get_params={
					"code": code
				},
				output_class=str
			)

		def _filter_response(self, response):
			return f"{response.get('hosts')[0]}{response.get('path')}".replace('\/', '/')

	def __init__(self, token, folder, pcloud_base_url="https://api.pcloud.com/"):
		self.__base_path = folder
		self.__client = PCloudClient.PCloudNetworkClient(token=token, url=pcloud_base_url)

	def __get_complete_path(self, path: str) -> str:
		return os.path.normpath(os.path.join(self.__base_path, path))

	def get_url(self, path) -> str:
		code = self.__client.execute(
			PCloudClient.CreateUrlRequest(
				self.__get_complete_path(path)
			)
		)
		return self.__client.execute(
			PCloudClient.GetUrlRequest(
				code
			)
		)

	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		if upload_path is None:
			upload_path = ""
		print(f"[+]Uploading {file_path} => {self.__get_complete_path(upload_path)}")
		self.__client.execute(
			PCloudClient.UploadFileRequest(
				file_path,
				self.__get_complete_path(upload_path)
			)
		)