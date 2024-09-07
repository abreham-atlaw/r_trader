import typing
from datetime import datetime

import numpy as np

import os


class FileSizeCleaner:

	def __init__(self, size: int = None, accumulation_path: str = None, accumulation_mode: bool = True):
		self.__size = size
		self.__accumulation_path = accumulation_path
		self.__accumulation: np.ndarray = None
		self.__accumulation_mode = accumulation_mode
		self.__accumulated_files = []

	def _generate_save_path(self) -> str:
		return os.path.join(self.__accumulation_path, f"{datetime.now().timestamp()}.npy")

	@staticmethod
	def __get_size(files: typing.List[str]) -> int:
		sizes = [np.load(file).shape[0] for file in files]
		return np.min(sizes)

	def __checkpoint_accumulation(self):
		if self.__accumulation.shape[0] < self.__size:
			return
		savable = self.__accumulation[:self.__size]
		self.__accumulation = self.__accumulation[self.__size:]
		path = self._generate_save_path()
		print(f"[+]Saving Accumulation {path}")
		np.save(path, savable)
		self.__accumulated_files.append(path)

	def __accumulate(self, array: np.ndarray):
		if not self.__accumulation_mode:
			return
		if self.__accumulation is None:
			self.__accumulation = array
		else:
			self.__accumulation = np.concatenate((self.__accumulation, array))
		self.__checkpoint_accumulation()

	def start(self, files: typing.List[str], verbose=2):
		if verbose > 0:
			print(f"[+]Processing {len(files)} Files...")
		if self.__size is None:
			self.__size = self.__get_size(files)
		size = self.__size
		if verbose > 0:
			print(f"[+]Using Size {size}")
		removed_files = []
		updated_files = []
		clean_files = []
		for file in files:
			arr = np.load(file)
			file_size = arr.shape[0]
			if file_size == size:
				clean_files.append(file)
				continue
			if verbose > 1:
				print(f"\n\n[+]Irregular File Found: {file}(size: {file_size})")
			if file_size > size:
				if verbose > 1:
					print(f"Updating {file}")
				np.save(file, arr[:size])
				self.__accumulate(arr[size:])
				updated_files.append(file)
			else:
				if verbose > 1:
					print(f"Removing {file}")
				os.system(f"rm {os.path.abspath(file)}")
				self.__accumulate(arr)
				removed_files.append(file)
		if verbose > 0:
			print(f"Removed Files: {len(removed_files)}")
			print(f"Updated Files: {len(updated_files)}")
			print(f"Clean Files: {len(clean_files)}")
			print(f"Accumulated Files: {len(self.__accumulated_files)}")

		return removed_files, updated_files, clean_files
