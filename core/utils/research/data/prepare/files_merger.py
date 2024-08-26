import os

import numpy as np


class FilesMerger:

	def __init__(self, x_size: int = None):
		self.__x_size = x_size

	def merge(self, x_path: str, y_path: str, out_path: str):
		print(f"Merging {x_path} and {y_path} into {out_path}")
		filenames = os.listdir(x_path)
		for i, filename in enumerate(filenames):
			x, y = [np.load(os.path.join(container, filename)) for container in [x_path, y_path]]
			if self.__x_size is None:
				self.__x_size = x.shape[1]
				print(f"Set X Size to {self.__x_size}")
			np.save(os.path.join(out_path, filename), np.concatenate((x, y), axis=1))
			print(f"Completed {(i+1)*100/len(filenames) :.2f}%...", end="\r")

	def split(self, x_path: str, y_path: str, in_path: str):
		print(f"Splitting {x_path} and {y_path} from {in_path}")
		filenames = os.listdir(in_path)
		for i, filename in enumerate(filenames):
			array = np.load(os.path.join(in_path, filename))
			x, y = array[:, :self.__x_size], array[:, self.__x_size:]
			for arr, path in zip((x, y), (x_path, y_path)):
				np.save(os.path.join(path, filename), arr)
			print(f"Completed {(i+1)*100/len(filenames) :.2f}%...", end="\r")
