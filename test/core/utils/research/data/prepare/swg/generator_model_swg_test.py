import unittest

from core import Config
from core.utils.research.data.prepare.swg.swg.generator_model_swg import GeneratorModelSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg.training import SampleWeightGenerationModel


class GeneratorModelSampleWeightGeneratorTest(unittest.TestCase):

	def setUp(self):
		data_path = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test",

		self.generator = GeneratorModelSampleWeightGenerator(

			model=SampleWeightGenerationModel.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/xgboost_swg_model.xgb"),
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w",
			X_extra_len=124,
			y_extra_len=1,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)

	def test_functionality(self):
		self.generator.start()
