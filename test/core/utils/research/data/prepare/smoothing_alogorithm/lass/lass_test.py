import os
import random
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import Lass, MovingAverage


class LassTest(unittest.TestCase):

	def setUp(self):
		self.lass = Lass("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-lass-training-cnn-0-it-0-tot.zip")
		self.df = pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-50k.csv"))
		self.sequence = self.df["c"].to_numpy()[-int(1e5):]
		self.gran = 1

	def test_functionality(self):
		y = self.lass.apply(self.sequence[::self.gran])
		self.assertIsInstance(y, np.ndarray)
		print(y)

	def test_plot_output(self):
		ma = MovingAverage(32)

		x = self.sequence[::self.gran]
		y = x
		for i in range(1):
			y = self.lass(y)
		y_ma = ma(x)

		x, y_ma = [
			arr[-y.shape[0]:]
			for arr in [x, y_ma]
		]

		view_window = 512
		samples = 6
		cols = 3
		rows = int(np.ceil(samples/cols))

		idxs = np.random.randint(0, x.shape[0]-view_window, samples)

		plt.figure()
		for i, j in enumerate(idxs):
			plt.subplot(rows, cols, i+1)
			plt.grid(True)
			for arr in [x, y, y_ma]:
				plt.plot(arr[j: j+view_window])

		plt.show()
