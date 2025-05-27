from torch import nn as nn

from core.utils.research.data.load import BaseDataset
from core.utils.research.utils.model_analysis.model_analyzer import ModelAnalyzer
from core.utils.research.utils.model_analysis.model_analyzer.mean_layer_output_ma import MeanLayerOutputModelAnalyzer
from test.core.utils.research.utils.model_analysis.model_analyzer.abstract_model_analyzer_test import \
	AbstractModelAnalyzerTest


class MeanLayerOutputModelAnalyzerTest(AbstractModelAnalyzerTest):

	def _init_analyzer(self, model: nn.Module, dataset: BaseDataset) -> ModelAnalyzer:
		return MeanLayerOutputModelAnalyzer(
			model=model,
			dataset=dataset,
			layers={
				"Layer 0": model.layers[0],
				"Layer 1": model.layers[1]
			}
		)
