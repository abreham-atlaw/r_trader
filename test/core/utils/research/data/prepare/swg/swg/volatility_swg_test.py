import unittest

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg import VolatilitySampleWeightGenerator
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest


class VolatilitySampleWeightGeneratorTest(BasicSampleWeightGeneratorTest ,unittest.TestCase):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return VolatilitySampleWeightGenerator(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			p=48
		)

