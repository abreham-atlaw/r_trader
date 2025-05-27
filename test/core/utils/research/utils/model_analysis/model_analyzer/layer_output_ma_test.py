import unittest

from torch import nn as nn

from core.utils.research.data.load import BaseDataset
from core.utils.research.utils.model_analysis.model_analyzer import ModelAnalyzer
from core.utils.research.utils.model_analysis.model_analyzer.layer_output_ma import LayerOutputModelAnalyzer

from test.core.utils.research.utils.model_analysis.model_analyzer.abstract_model_analyzer_test import \
	AbstractModelAnalyzerTest


class LayerOutputModelAnalyzerTest(AbstractModelAnalyzerTest):

	def _init_analyzer(self, model: nn.Module, dataset: BaseDataset) -> ModelAnalyzer:
		return LayerOutputModelAnalyzer(
			model=model,
			dataset=dataset,
			layers={
				"Layer 0": model.layers[0],
				"Layer 1": model.layers[1]
			}
		)
