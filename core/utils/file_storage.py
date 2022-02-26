from typing import *

import dropbox

import os

from core import Config


class DropboxClient:

	def __init__(self, token=Config.DROPBOX_API_TOKEN, folder=Config.DROPBOX_FOLDER):
		self.__client = dropbox.Dropbox(token)
		self.__folder = folder

	def upload_file(self, file_path: str, upload_path: Union[str, None] = None):
		if upload_path is None:
			upload_path = os.path.join(self.__folder, file_path.split("/")[-1])

		with open(file_path, "rb") as f:
			meta = self.__client.files_upload(f.read(), upload_path, mode=dropbox.files.WriteMode("overwrite"))
			return meta
