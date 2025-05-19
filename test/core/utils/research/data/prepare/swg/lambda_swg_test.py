import unittest

import numpy as np

from core import Config
from core.utils.research.data.prepare.swg import LambdaSampleWeightGenerator


class LambdaSampleWeightGeneratorTest(unittest.TestCase):

	def setUp(self):
		self.generator = LambdaSampleWeightGenerator(
			data_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test",
			export_path="/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/4/test/w",
			func=lambda X, y: ((np.abs((y - X[:, -1]) - (X[:, -1] - X[:, -2])) * 3e4) ** (1/48) * 1/0.95)**24,
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		)

	def test_functionality(self):
		self.generator.start()
