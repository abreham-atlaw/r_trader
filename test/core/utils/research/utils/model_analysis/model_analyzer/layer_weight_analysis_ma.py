import unittest

from torch import nn as nn

from core.utils.research.data.load import BaseDataset
from core.utils.research.utils.model_analysis.model_analyzer import ModelAnalyzer
from core.utils.research.utils.model_analysis.model_analyzer.layer_weight_ma import LayerWeightModelAnalyzer

from test.core.utils.research.utils.model_analysis.model_analyzer.abstract_model_analyzer_test import \
	AbstractModelAnalyzerTest


class LayerWeightAnalysisModelAnalyzerTest(AbstractModelAnalyzerTest):

	def _init_analyzer(self, model: nn.Module, dataset: BaseDataset) -> ModelAnalyzer:
		return LayerWeightModelAnalyzer(
			model=model,
			dataset=dataset
		)
