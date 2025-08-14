import pandas as pd

import os
import unittest

from core import Config
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass_preparer_2 import LassPreparer2
from core.utils.research.data.prepare.splitting import SequentialSplitter


class LassPreparer2Test(unittest.TestCase):

	def setUp(self):
		self.preparer = LassPreparer2(
			sa=MovingAverage(window_size=32),
			shift=16,
			block_size=32,
			granularity=30,
			batch_size=64,
			output_path=os.path.join(Config.BASE_DIR, "temp/Data/lass/1"),
			order_gran=True,
			df=pd.read_csv(os.path.join(Config.BASE_DIR, "temp/Data/AUD-USD-50k.csv")),
			splitter=SequentialSplitter(test_size=0.2)
		)

	def test_functionality(self):
		self.preparer.start()
