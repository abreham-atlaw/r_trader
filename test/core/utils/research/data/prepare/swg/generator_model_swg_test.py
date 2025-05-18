import unittest

from core import Config
from core.utils.research.data.prepare.swg.generator_model_swg import GeneratorModelSampleWeightGenerator
from core.utils.research.data.prepare.swg.training.models import SampleWeightGenerationModel


class GeneratorModelSampleWeightGeneratorTest(unittest.TestCase):

	def setUp(self):
		self.generator = GeneratorModelSampleWeightGenerator(
			data_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test",
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w",
			model=SampleWeightGenerationModel.load("/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/xgboost_swg_model.xgb"),
			X_extra_len=124,
			y_extra_len=1,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)

	def test_functionality(self):
		self.generator.start()
