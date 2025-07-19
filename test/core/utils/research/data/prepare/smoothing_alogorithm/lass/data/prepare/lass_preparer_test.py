import os
import unittest

import pandas as pd

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass_preparer import LassPreparer


class LassPreparerTest(unittest.TestCase):

	def setUp(self):
		self.preparer = LassPreparer(
			sa=MovingAverage(window_size=32),
			shift=16,
			block_size=32,
			granularity=30,
			batch_size=64,
			output_path=os.path.join(Config.BASE_DIR, "temp/Data/lass/0"),
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-50k.csv"))
		)

	def test_functionality(self):
		self.preparer.start()
