import unittest

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg import IdealModelSampleWeightGenerator
from core.utils.research.losses import ProximalMaskedLoss
from lib.utils.torch_utils.model_handler import ModelHandler
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest


class IdealModelSampleWeightGeneratorTest(BasicSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return IdealModelSampleWeightGenerator(
			loss=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				collapsed=False
			),
			model=ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/1740917588.839826.zip"),
		)
