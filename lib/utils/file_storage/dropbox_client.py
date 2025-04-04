import typing
from typing import *

import dropbox

import os

from .file_storage import FileStorage, MetaData


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

	def mkdir(self, path: str):
		pass

	def get_metadata_raw(self, path: str) -> typing.Dict[str, typing.Any]:
		pass

	def get_metadata(self, path: str) -> MetaData:
		pass
