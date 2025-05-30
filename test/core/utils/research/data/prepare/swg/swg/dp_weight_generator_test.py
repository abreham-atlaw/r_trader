import unittest

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg import DeltaSampleWeightGenerator
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest


class DeltaSampleWeightGeneratorTest(BasicSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return DeltaSampleWeightGenerator(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			alpha=8
		)
