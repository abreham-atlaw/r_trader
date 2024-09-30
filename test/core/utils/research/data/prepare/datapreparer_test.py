import json
import os
import unittest
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.utils.research.data.prepare.datapreparer import DataPreparer


class DataPreparerTest(unittest.TestCase):

	def __generate_data(self, path: str, size: int, start=0, end=100):
		print("[+]Generating...")
		sequence = np.linspace(start, end, size)
		sequence = np.sin(sequence) + 1.5
		df = pd.DataFrame(columns=["c"])
		df["c"] = sequence
		df.to_csv(path)

	def test_functionality(self):

		BOUNDS = [0.5, 0.75, 1, 1.25, 1.5]
		BLOCK_SIZE = 3
		GRAN = 2
		MOVING_AVERAGE = None
		df = pd.DataFrame(data={
			"c": np.array([2, 3, 4, 5, 6, 9, 12, 3, 15, 3])
		})
		EXPECTED_OUTPUT = (
			np.array([
				[2, 4, 6],
				[4, 6, 12],
				[3, 5, 9],
				[5, 9, 3]
			]),
			np.array([
				[0, 0, 0, 0, 0, 1],
				[0, 0, 0, 0, 1, 0],
				[1, 0, 0, 0, 0, 0],
				[0, 0, 0, 1, 0, 0]
			])
		)

		datapreparer = DataPreparer(
			boundaries=BOUNDS,
			block_size=BLOCK_SIZE,
			ma_window_size=MOVING_AVERAGE,
			test_split_size=0.25,
			granularity=GRAN
		)

		datapreparer.start(
			df=df,
			save_path="/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared"
		)

	def test_actual(self):
		path = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-10k.csv"
		# self.__generate_data(path, int(5e4))
		df = pd.read_csv(path)
		GRAN = 5
		MOVING_AVERAGE = 10
		BLOCK_SIZE = 1024
		BOUNDS = sorted(list(np.concatenate([
			1 + bound * np.linspace(-1, 1, size)**pow
			for bound, size, pow in [
				(4e-3, 64, 3),
				(1e-4, 128, 3),
				(2e-4, 128, 3),
				(3e-4, 128, 3)
			]
		])))
		BATCH_SIZE = int(1e3)
		TEST_SPLIT_SIZE = 0.25

		datapreparer = DataPreparer(
			boundaries=BOUNDS,
			block_size=BLOCK_SIZE,
			ma_window_size=MOVING_AVERAGE,
			test_split_size=TEST_SPLIT_SIZE,
			granularity=GRAN,
			batch_size=BATCH_SIZE,
		)

		datapreparer.start(
			df=df,
			save_path="/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared_actual",
			export_remaining=False
		)

	def __generate_bounds(self, n, mn, mx, p):
		f = lambda x: (((mx-1)/((n/2)**p)) * ((x - (n/2)) ** p)) + 1

		return np.array([f(x) for x in range(n)])

	def test_plot_frequencies(self):

		path = "/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/preprared(plot)"

		THRESHOLD = 500

		# BOUNDS = sorted(list(np.concatenate([
		# 	1 + bound * np.linspace(-1, 1, size) ** pow
		# 	for bound, size, pow in [
		# 		(4e-3, 64, 3),
		# 		(1e-4, 128, 3),
		# 		(2e-4, 128, 3),
		# 		(3e-4, 128, 3)
		# 	]
		# ])))
		# BOUNDS = sorted(list(np.concatenate([
		# 	1 + bound * np.linspace(-1, 1, size) ** pow
		# 	for bound, size, pow in [
		# 		(4e-3, 64, 3),
		# 		(1e-4, 128, 3),
		# 		(2e-4, 128, 3),
		# 		(3e-4, 128, 3)
		# 	]
		# ])))
		# BOUNDS += sorted(list(np.linspace(0.9724920650187554, 1.0328659854109465, 1000)))
		#
		# BOUNDS = sorted(list(BOUNDS))
		# BOUNDS = np.linspace(BOUNDS[100], BOUNDS[250], 2)
		# BOUNDS = np.linspace(0.9724920650187554, 1.0328659854109465, 100)

		BOUNDS = sorted(list(np.concatenate([
			self.__generate_bounds(
				n=7,
				mn=0.9527526705012326,
				mx=1.0336048383163439,
				p=p
			)
			for p in [(2*n)+1 for n in range(10)]
			# for n in range(3, 7)
		])))

		with open("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/bounds/1727643313.914046.json", 'r') as f:
			BOUNDS = json.load(f)

		datapreparer = DataPreparer(
			boundaries=BOUNDS,
			block_size=20,
			ma_window_size=10,
			test_split_size=0.1,
			granularity=5,
			batch_size=int(1e9),
		)
		df = pd.read_csv("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-50k.csv")

		datapreparer.start(
			df=df,
			save_path=path,
			export_remaining=True
		)

		file = os.path.join(path, "train/y", sorted(os.listdir(os.path.join(path, "train/y")))[-1])
		y = np.load(file)
		y_classes = np.argmax(y, axis=1)

		classes, frequencies = np.unique(y_classes, return_counts=True)

		print(f"Classes: {len(classes)}")
		print(f"Efficiency: {len(classes)/(len(BOUNDS)+1)}")

		valid_bounds = [BOUNDS[idx] for idx in classes if idx < len(BOUNDS) and (THRESHOLD is None or frequencies[idx] > THRESHOLD)]
		with open(f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/bounds/{datetime.now().timestamp()}.json", "w") as file:
			json.dump(valid_bounds, file)

		plt.scatter(classes, frequencies)

		plt.figure()
		plt.scatter(list(range(len(valid_bounds))), valid_bounds)

		plt.show()
