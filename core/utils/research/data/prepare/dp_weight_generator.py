import typing

import numpy as np

import os

from lib.utils.logger import Logger


class DPWeightGenerator:

	def __init__(
			self,
			data_path: str,
			export_path: str,
			bounds: typing.Union[typing.List[float], np.ndarray],
			alpha: float = 7,
			X_dir: str = "X",
			y_dir: str = "y",
			X_extra_len: int = 124,
			y_extra_len: int = 1
	):
		self.__data_path = data_path
		self.__export_path = export_path
		self.__X_dir = X_dir
		self.__y_dir = y_dir
		self.__alpha = alpha
		self.__X_extra_len = X_extra_len
		self.__y_extra_len = y_extra_len

		if isinstance(bounds, list):
			bounds = np.array(bounds)
		self.__bounds = bounds

	def __setup(self):
		os.makedirs(self.__export_path, exist_ok=True)

	def __generate_weights(self, X: np.ndarray, y: np.ndarray):
		X, y = [arr[:, :-extra_len] for arr, extra_len in zip([X, y], [self.__X_extra_len, self.__y_extra_len])]
		y = self.__bounds[np.argmax(y, axis=1)]
		return 10**(((((X[:, -1]/X[:, -2]) - 1) * (y - 1))*-1) * 10 ** self.__alpha)

	def __export_weights(self, weights: np.ndarray, filename: str):
		np.save(os.path.join(self.__export_path, filename), weights)

	def start(self):
		filenames = os.listdir(os.path.join(self.__data_path, self.__X_dir))

		for i, filename in enumerate(filenames):
			X, y = [
				np.load(os.path.join(self.__data_path, dir_name, filename))
				for dir_name in [self.__X_dir, self.__y_dir]
			]
			weights = self.__generate_weights(X, y)
			self.__export_weights(weights, filename)

			Logger.info(f"[+]Processed {(i+1)*100/len(filenames):.2f}%", end="\r")

		Logger.info(f"Done!")
