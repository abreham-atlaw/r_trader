import unittest

from torch import nn as nn

from core.utils.research.data.load import BaseDataset
from core.utils.research.utils.model_analysis.model_analyzer import ModelAnalyzer, InputSaliencyModelAnalyzer

from test.core.utils.research.utils.model_analysis.model_analyzer.abstract_model_analyzer_test import \
	AbstractModelAnalyzerTest


class InputSaliencyModelAnalyzerTest(AbstractModelAnalyzerTest):

	def _init_analyzer(self, model: nn.Module, dataset: BaseDataset) -> ModelAnalyzer:
		return InputSaliencyModelAnalyzer(model=model, dataset=dataset, sample_size=10)

