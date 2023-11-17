import unittest

from core.utils.training.datapreparation import CombinedDataPreparer
from core.utils.training.datapreparation.filters import MovingAverageFilter


class CombinedDataPreparerTest(unittest.TestCase):

	def test_functionality(self):
		preparer = CombinedDataPreparer(
			64,
			files=["/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/AUD-USD-50k.csv"],
			output_path='/home/abreham/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared',
			filters=[
				MovingAverageFilter(10)
			],
			export_remaining=True
		)
		preparer.start()
