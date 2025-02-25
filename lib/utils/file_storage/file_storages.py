import typing
from typing import *
from abc import ABC, abstractmethod

import dropbox

import random
import os
from threading import Thread

from lib.network.rest_interface.requests import Request
from lib.network.rest_interface.NetworkApiClient import NetworkApiClient
from .exceptions import FileNotFoundException, FileSystemException
from ..logger import Logger


class FileStorage(ABC):

	@abstractmethod
	def get_url(self, path) -> str:
		pass

	@abstractmethod
	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		pass

	@abstractmethod
	def listdir(self, path: str) -> typing.List[str]:
		pass

	@abstractmethod
	def delete(self, path: str):
		pass

	def exists(self, path: str) -> bool:
		try:
			files = self.listdir(os.path.dirname(path))
			return os.path.basename(path) in files
		except FileNotFoundException:
			return False

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
		path = os.path.join(self.__folder, path)
		links = self.__client.sharing_list_shared_links(path).links
		if len(links) > 0:
			url = links[0].url
		else:
			url = self.__client.sharing_create_shared_link_with_settings(path).url
		return f"{url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')}&raw=1"

	def listdir(self, path: str) -> typing.List[str]:
		res = self.__client.files_list_folder(path)
		return [entry.name for entry in res.entries]

	def delete(self, path: str):
		pass


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
			if isinstance(response, dict) and response.get("result") == 7001:
				raise FileNotFoundException()
			return f"{response.get('hosts')[0]}{response.get('path')}".replace('\/', '/')

	class ListDirRequest(Request):

		def __init__(self, path: str):
			super().__init__(
				"/listfolder",
				method=Request.Method.GET,
				get_params={
					"path": path
				},
				output_class=typing.List[str]
			)

		def _filter_response(self, response):
			return response["metadata"]["contents"]

		def deserialize_object(self, response) -> object:
			return [
				content["path"]
				for content in self._filter_response(response)
			]

	class DeleteFileRequest(Request):

		def __init__(self, path: str):
			super().__init__(
				"/deletefile",
				method=Request.Method.POST,
				get_params={
					"path": path
				}
			)
			self.__path = path

		def _filter_response(self, response):
			if response["result"] == 0:
				return response

			if response["result"] == 2009:
				raise FileNotFoundException()

			raise FileSystemException(f"Failed to delete file of path={self.__path}. {response}")

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
		Logger.info(f"Uploading {file_path} => {self.__get_complete_path(upload_path)}")
		self.__client.execute(
			PCloudClient.UploadFileRequest(
				file_path,
				self.__get_complete_path(upload_path)
			)
		)

	def listdir(self, path: str) -> typing.List[str]:
		try:
			return self.__client.execute(
				PCloudClient.ListDirRequest(
					self.__get_complete_path(path)
				)
			)
		except KeyError as ex:
			raise FileNotFoundException()

	def delete(self, path: str):
		self.__client.execute(
			PCloudClient.DeleteFileRequest(
				self.__get_complete_path(path)
			)
		)

class LocalStorage(FileStorage):

	class ServerThread(Thread):

		def __init__(self, dir: str, port: int, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.__dir, self.__port = dir, port

		def run(self) -> None:
			os.system(f"python3 -m http.server --directory {self.__dir}") #  TODO: Add port

	def __init__(self, base_path: str, port: Optional[int]=None):
		self.__base_path = os.path.abspath(base_path)
		self.__port = port
		if port is None:
			self.__port = random.randint(8000, 8999)
		self.__server = self.__start_server()

	def __start_server(self) -> 'LocalStorage.ServerThread':
		thread = LocalStorage.ServerThread(self.__base_path, self.__port)
		thread.start()
		return thread

	def get_url(self, path) -> str:

		return os.path.join(
			f"http://localhost:{self.__port}/",
			path
		)

	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		if upload_path is None:
			upload_path = ""
		os.system(f"cp {file_path} {os.path.join(self.__base_path, upload_path)}")

