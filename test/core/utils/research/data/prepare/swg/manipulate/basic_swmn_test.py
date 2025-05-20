import unittest

from core.utils.research.data.prepare.swg.manipulate import BasicSampleWeightManipulator


class BasicSampleWeightManipulatorTest(unittest.TestCase):

	def setUp(self):
		self.manipulator = BasicSampleWeightManipulator(
			weights_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w",
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w_manipulated",
			shift=3,
			scale=2,
			stretch=10
		)

	def test_functionality(self):
		self.manipulator.start()
