from torch import nn as nn

from core.utils.research.data.load import BaseDataset
from core.utils.research.utils.model_analysis.model_analyzer import ModelAnalyzer
from core.utils.research.utils.model_analysis.model_analyzer.layer_saliency_ma import LayerSaliencyModelAnalyzer
from test.core.utils.research.utils.model_analysis.model_analyzer.abstract_model_analyzer_test import \
	AbstractModelAnalyzerTest


class LayerSaliencyModelAnalyzerTest(AbstractModelAnalyzerTest):

	def _init_analyzer(self, model: nn.Module, dataset: BaseDataset) -> ModelAnalyzer:
		return LayerSaliencyModelAnalyzer(
			model=model,
			dataset=dataset,
			sample_size=10,
		)
