import unittest

import numpy as np
import pandas as pd

from core.utils.research.data.prepare.datapreparer import DataPreparer


class DataPreparerTest(unittest.TestCase):

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
		df = pd.read_csv("/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-5k.csv")
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
		BATCH_SIZE = int(5e3)
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
			save_path="/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared",
			export_remaining=False
		)
