import os
from datetime import datetime

import numpy as np

from lib.utils.logger import Logger


class BatchSizeModifier:

	def __init__(
			self,
			source_path: str,
			target_path: str,
			target_batch_size: int,
			X_dir: str = "X",
			y_dir: str = "y"
	):
		self.__source_path: str = source_path
		self.__target_path: str = target_path
		self.__target_batch_size = target_batch_size
		self.__X_dir, self.__y_dir = X_dir, y_dir
		self.__X, self.__y = None, None
		self.__setup_dirs()

	def __setup_dirs(self):
		for dir_ in self.__X_dir, self.__y_dir:
			path = os.path.join(self.__target_path, dir_)
			if not os.path.exists(path):
				Logger.info(f"[+]Creating {path}...")
				os.makedirs(path)

	@staticmethod
	def __generate_filename():
		return f"{datetime.now().timestamp()}.npy"

	def __dump(self, X: np.ndarray, y: np.ndarray):

		filename = self.__generate_filename()

		for arr, dir_ in zip([X, y], [self.__X_dir, self.__y_dir]):
			np.save(os.path.join(self.__target_path, dir_, filename), arr)

	def __store(self, X: np.ndarray, y: np.ndarray):
		if self.__X is None:
			self.__X = X
			self.__y = y
		else:
			self.__X = np.concatenate([self.__X, X])
			self.__y = np.concatenate([self.__y, y])
		if self.__X.shape[0] >= self.__target_batch_size:
			X, y = [arr[:self.__target_batch_size] for arr in [self.__X, self.__y]]
			self.__dump(X, y)
			self.__X, self.__y = [arr[self.__target_batch_size:] for arr in [self.__X, self.__y]]

	def __process_file(self, filename: str):
		X, y = [
			np.load(os.path.join(self.__source_path, dir_, filename))
			for dir_ in [self.__X_dir, self.__y_dir]
		]
		self.__store(X, y)

	def start(self):
		Logger.info(f"[+]Modifying Batch Size to {self.__target_batch_size}")

		files = sorted(os.listdir(os.path.join(self.__source_path, self.__X_dir)))

		for i, filename in enumerate(files):
			self.__process_file(filename)
			Logger.info(f"[+]Processed {i+1}/{len(files)} files ...", end="\r")
