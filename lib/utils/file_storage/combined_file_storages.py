import os.path
import random
import typing

from .file_storages import FileStorage, PCloudClient
from .exceptions import FileNotFoundException


class CombinedFileStorage(FileStorage):

	def __init__(self, children: typing.List[FileStorage]):
		self.__children = children

	def _get_storage(self, path) -> FileStorage:
		for child in self.__children:
			try:
				child.get_url(path)
				return child
			except FileNotFoundException:
				pass
		raise FileNotFoundException

	def _choose_storage(self, file_path: str, upload_path: typing.Union[str, None] = None):
		try:
			path = upload_path
			if upload_path is None:
				path = os.path.basename(file_path)
			return self._get_storage(path)
		except FileNotFoundException:
			return random.choice(self.__children)

	def get_url(self, path) -> str:
		storage = self._get_storage(path)
		return storage.get_url(path)

	def upload_file(self, file_path: str, upload_path: typing.Union[str, None] = None):
		storage = self._choose_storage(file_path, upload_path)
		storage.upload_file(file_path, upload_path)

	def listdir(self, path: str) -> typing.List[str]:
		files = []
		for child in self.__children:
			try:
				files += child.listdir(path)
			except FileNotFoundException:
				pass
		return files


class PCloudCombinedFileStorage(CombinedFileStorage):

	def __init__(self, tokens: typing.List[str], base_path: str):
		super().__init__(
			[
				PCloudClient(token, base_path)
				for token in tokens
			]
		)
