import unittest

import numpy as np

from core.utils.training.datapreparation.datapreparer.gran_datapreparer import GranDataPreparer
from core.utils.training.datapreparation.filters import MovingAverageFilter


class GranDataPreparerTest(unittest.TestCase):

	def test_functionality(self):
		preparer = GranDataPreparer(
			seq_len=64,
			percentages=sorted(list(1 + 4e-3 * np.linspace(-1, 1, 64)**3) + list(1 + 1e-4 * np.linspace(-1, 1, 128)**3) + list(1 + 2e-4 * np.linspace(-1, 1, 128)**3) + list(1 + 3e-4 * np.linspace(-1, 1, 128)**3)),
			files=["/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-5k.csv"],
			output_path='/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared',
			filters=[
				MovingAverageFilter(10)
			],
			export_remaining=True
		)
		preparer.start()
