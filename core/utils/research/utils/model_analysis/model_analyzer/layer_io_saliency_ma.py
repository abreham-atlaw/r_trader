import typing

import torch
from torch import nn

from .model_analyzer import ModelAnalyzer
from .utils.layer_utils import LayerUtils
from ..f import get_layer_io_saliency
from .abstract_layer_ma import AbstractLayerModelAnalyzer


class LayerIOSaliencyModelAnalyzer(AbstractLayerModelAnalyzer):

	def _get_analysis(
			self,
			model: nn.Module,
			layer: nn.Module,
			X: torch.Tensor,
			y: torch.Tensor,
			w: torch.Tensor
	) -> torch.Tensor:
		return get_layer_io_saliency(model, X, layer)
