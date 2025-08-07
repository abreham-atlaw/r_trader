import unittest
from abc import ABC, abstractmethod

import torch.nn as nn

from core.utils.research.data.load import BaseDataset
from core.utils.research.utils.model_analysis.model_analyzer import ModelAnalyzer
from lib.utils.torch_utils.model_handler import ModelHandler


class AbstractModelAnalyzerTest(unittest.TestCase, ABC):

	_MODEL_PATH = "/home/abrehamatlaw/Downloads/Compressed/abrehamalemu-spinoza-training-cnn-33-it-42-tot.zip"
	_DATA_PATH = "/home/abrehamatlaw/Projects/PersonalProjects/RTrader/r_trader/temp/Data/prepared/7/train"

	@abstractmethod
	def _init_analyzer(self, model: nn.Module, dataset: BaseDataset) -> ModelAnalyzer:
		pass

	def setUp(self):
		self.model = ModelHandler.load(self._MODEL_PATH)
		self.dataset = BaseDataset(
			root_dirs=[self._DATA_PATH]
		)

		self.analyzer = self._init_analyzer(self.model, self.dataset)

	def test_functionality(self):
		self.analyzer.start()
