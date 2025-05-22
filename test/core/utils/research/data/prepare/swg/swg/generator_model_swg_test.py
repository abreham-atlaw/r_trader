import unittest

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg.generator_model_swg import GeneratorModelSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg.training.models import SampleWeightGenerationModel
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest


class GeneratorModelSampleWeightGeneratorTest(BasicSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return GeneratorModelSampleWeightGenerator(
			model=SampleWeightGenerationModel.load(
				"/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/models/xgboost_swg_model.xgb"),
			X_extra_len=124,
			y_extra_len=1,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)
