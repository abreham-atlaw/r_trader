import typing

import numpy as np
import torch
from torch import nn as nn

from .abstract_layer_ma import AbstractLayerModelAnalyzer
from .utils.layer_utils import LayerUtils
from .utils.plot_utils import PlotUtils


class LayerWeightModelAnalyzer(AbstractLayerModelAnalyzer):

	def _get_analysis(
			self, model: nn.Module, layer: nn.Module, X: torch.Tensor, y: torch.Tensor,
			w: torch.Tensor
	) -> typing.Dict[str, torch.Tensor]:
		return LayerUtils.get_layer_weights(layer)

	def _generate_name(self, layer_name: str, param_name: str) -> str:
		return f"{layer_name}.{param_name}"

	def _export_layer_analysis(self, name: str, layer: nn.Module, analysis: typing.Dict[str, torch.Tensor]):

		for param_name, param in analysis.items():
			super()._export_layer_analysis(self._generate_name(name, param_name), layer, param)

	def _plot_layer_analysis(self, name: str, layer: nn.Module, analysis: typing.Dict[str, torch.Tensor]):

		for param_name, param in analysis.items():
			super()._plot_layer_analysis(self._generate_name(name, param_name), layer, param)

	def _plot_analysis(self, analysis: torch.Tensor, title: str):
		PlotUtils.plot(y=analysis, title=title, mode=PlotUtils.Mode.IMAGE)
