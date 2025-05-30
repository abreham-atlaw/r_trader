
import torch
from torch import nn


from .abstract_layer_ma import AbstractLayerModelAnalyzer
from core.utils.research.utils.model_analysis.f import get_layer_saliency


class LayerSaliencyModelAnalyzer(AbstractLayerModelAnalyzer):

	def _get_analysis(
			self,
			model: nn.Module,
			layer: nn.Module,
			X: torch.Tensor,
			y: torch.Tensor,
			w: torch.Tensor
	) -> torch.Tensor:
		saliency = get_layer_saliency(model, X, layer)
		return saliency
