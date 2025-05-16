import unittest

from core.utils.research.data.prepare.swg.merge import MeanSampleWeightMerger


class MeanSampleWeightMergerTest(unittest.TestCase):

	def setUp(self):
		self.merger = MeanSampleWeightMerger(
			weight_paths=[
				f"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/{i}/test/w"
				for i in [4, 5]
			],
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/3/train/w"
		)

	def test_functionality(self):
		self.merger.start()
