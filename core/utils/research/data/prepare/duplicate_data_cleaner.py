import typing

import numpy as np

from core.utils.research.data.prepare.duplicate_cleaner import DuplicateCleaner


class DuplicateDataCleaner:

	def __init__(self, cleaner=None):
		if cleaner is None:
			cleaner = DuplicateCleaner()
		self.__cleaner = cleaner

	def __clean_file(self, X_file: str, y_file: str):
		X, y = self.__cleaner.clean(np.load(X_file), companion_array=np.load(y_file))
		for arr, path in zip((X, y), (X_file, y_file)):
			np.save(path, arr)

	def start(self, X_files: typing.List[str], y_files: typing.List[str] = None):
		for i, (X_file, y_file) in enumerate(zip(X_files, y_files)):
			self.__clean_file(X_file, y_file)
