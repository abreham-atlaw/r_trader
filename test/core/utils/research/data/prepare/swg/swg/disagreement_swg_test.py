import unittest

import numpy as np

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg import DisagreementSampleWeightGenerator
from core.utils.research.losses import ProximalMaskedLoss
from lib.utils.torch_utils.model_handler import ModelHandler
from test.core.utils.research.data.prepare.swg.abstract_swg_test import AbstractSampleWeightGeneratorTest
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest


class DisagreementSampleWeightGeneratorTest(BasicSampleWeightGeneratorTest, unittest.TestCase):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return DisagreementSampleWeightGenerator(
			loss=ProximalMaskedLoss(
				n=len(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND) + 1,
				collapsed=False
			),
			anchor_model=ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/1740917588.839826.zip"),
			weak_model=ModelHandler.load("/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-rtrader-training-exp-0-cnn-35-cum-0-it-27-tot.zip")
		)
