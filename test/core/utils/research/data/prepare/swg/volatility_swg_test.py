import unittest

from core import Config
from core.utils.research.data.prepare.swg import VolatilitySampleWeightGenerator


class VolatilitySampleWeightGeneratorTest(unittest.TestCase):

	def setUp(self):
		self.generator = VolatilitySampleWeightGenerator(
			data_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test",
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w",
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			p=48
		)

	def test_functionality(self):
		self.generator.start()
