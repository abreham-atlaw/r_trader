import os
import random
import uuid

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from core.utils.research.data.prepare import DataPreparer


class BoundGenerator:

	def __init__(
			self,
			start: float,
			end: float,
			csv_path: str,
			threshold=None,
			tmp_path="/tmp",
	):
		self.__start = start
		self.__end = end
		self.__threshold = threshold
		self.__df = pd.read_csv(csv_path)
		self.__tmp_path = tmp_path

	def __prepare_tmp_path(self):
		path = os.path.join(self.__tmp_path, f"{uuid.uuid4()}.bo")
		return path

	def __poly_generate(self, n):
		p = 2*np.random.randint(0, 100) + 1
		q = np.random.randint(n, n*10)
		f = lambda x: (((self.__end - 1) / ((q / 2) ** p)) * ((x - (q / 2)) ** p)) + 1
		return random.choices(np.array([f(x) for x in range(q)]), k=n)

	def __random_generate(self, n):
		return np.random.uniform(self.__start, self.__end, n)

	def __linear_generate(self, n):
		bound = [x + (s * ((self.__end - self.__start) / 2)) for s, x in zip([-1, 1], [self.__start, self.__end])]
		return np.linspace(*bound, n)

	def __generator(self, n):
		return random.choice([self.__poly_generate, self.__random_generate, self.__linear_generate])(n)

	def __get_frequencies(self, bounds):
		datapreparer = DataPreparer(
			boundaries=bounds,
			block_size=20,
			ma_window_size=10,
			test_split_size=0.1,
			granularity=5,
			batch_size=int(1e9),
			verbose=False
		)

		path = self.__prepare_tmp_path()
		datapreparer.start(
			df=self.__df,
			save_path=path,
			export_remaining=True
		)
		y = np.concatenate([
			np.load(os.path.join(path, "train/y", f))
			for f in sorted(os.listdir(os.path.join(path, "train/y")))
		])
		y_classes = np.argmax(y, axis=1)
		classes, frequencies = np.unique(y_classes, return_counts=True)
		frequencies = frequencies / np.sum(frequencies)
		os.system(f"rm -fr \"{path}\"")
		return classes, frequencies

	def __plot(self, bounds, indexes, frequencies):
		plt.close('all')

		plt.figure()
		plt.scatter(indexes, frequencies)

		plt.figure()
		plt.scatter(list(range(len(bounds))), bounds)

		plt.show()

	def __filter_valid(self, bounds, threshold=None, plot=False):
		indexes, frequencies = self.__get_frequencies(bounds)

		valid_bounds = [
			bounds[idx]
			for i, idx in enumerate(indexes)
			if idx < len(bounds) and (threshold is None or frequencies[i] > threshold)
		]

		if plot:
			self.__plot(valid_bounds, indexes, frequencies)

		return valid_bounds

	def __generate(self, n, bounds):
		bounds = sorted(bounds + list(self.__generator(n - len(bounds))))
		bounds = self.__filter_valid(bounds, threshold=self.__threshold)
		print(f"Found bounds: {len(bounds)}")
		if len(bounds) < n:
			bounds = self.__generate(n, bounds)
		return bounds

	def generate(self, n, plot=False):
		print(f"Generating {n} bounds...")
		bounds = self.__generate(n, [])

		if plot:
			self.__filter_valid(bounds, threshold=0, plot=True)

		return bounds
