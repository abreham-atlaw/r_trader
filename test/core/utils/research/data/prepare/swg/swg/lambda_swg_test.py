import unittest

import numpy as np

from core import Config
from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from core.utils.research.data.prepare.swg.swg import LambdaSampleWeightGenerator
from test.core.utils.research.data.prepare.swg.swg.basic_swg_test import BasicSampleWeightGeneratorTest


class LambdaSampleWeightGeneratorTest(BasicSampleWeightGeneratorTest):

	def _init_generator(self) -> AbstractSampleWeightGenerator:
		return LambdaSampleWeightGenerator(
			func=lambda X, y: ((np.abs((y - X[:, -1]) - (X[:, -1] - X[:, -2])) * 3e4) ** (1/48) * 1/0.95)**24,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)
