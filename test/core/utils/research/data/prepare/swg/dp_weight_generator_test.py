import unittest

from core import Config
from core.utils.research.data.prepare.swg import DeltaSampleWeightGenerator


class DeltaSampleWeightGeneratorTest(unittest.TestCase):

	def test_functionality(self):
		generator = DeltaSampleWeightGenerator(
			data_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test",
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/dp_weights/4/test",
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			alpha=8
		)
		generator.start()
