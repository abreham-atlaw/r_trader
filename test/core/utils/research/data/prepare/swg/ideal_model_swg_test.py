import unittest

from core import Config
from core.utils.research.data.prepare.swg import IdealModelSampleWeightGenerator
from core.utils.research.losses import ProximalMaskedLoss
from lib.utils.torch_utils.model_handler import ModelHandler


class IdealModelSampleWeightGeneratorTest(unittest.TestCase):

	def setUp(self):
		self.generator = IdealModelSampleWeightGenerator(
			data_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test",
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w",
			loss=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				collapsed=False
			),
			model=ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/1740917588.839826.zip"),
		)

	def test_functionality(self):
		self.generator.start()
