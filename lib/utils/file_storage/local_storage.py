from typing import *

import random
import os
from threading import Thread

from .file_storage import FileStorage


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

